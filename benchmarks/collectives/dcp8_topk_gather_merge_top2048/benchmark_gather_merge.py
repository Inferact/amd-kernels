# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank graph-replay time of the fused gather + merge against vLLM's stock sequence, with the roofline table.

Stock: AITER IPC all-gather of the fp32 scores + vllm merge_dcp_scores (AITER top-k values, Triton owned-slot
selection, Triton compaction); the table also gives the all-gather alone. Floors: BW = bytes pushed per rank (upper
bound: histogram + 512 keys per row and peer) over the measured all-to-all output rate of 343 GB/s; latency = launch
1.57 us + two dependent xGMI hops (data, then row counts) at 1.3 us each (0.8 one way + 0.5 poll) + 0.5 us loads
and stores + 2 us of selection.
Needs vLLM; run with eight visible GPUs. Graphs of 100 calls on 100 distinct inputs,
MAX over ranks, min of 15 rounds.
"""
import contextlib
import os
import socket
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from amd_kernels.collectives.dcp8_topk_gather_merge_top2048 import TopkGatherMerge
from amd_kernels.collectives.dcp8_topk_gather_merge_top2048.reference import reference

MAX_ROWS = 256
WORLD, K, PAGE, STRIDE, BLOCKS = 8, 2048, 64, 72, 64
XGMI_OUT_GBS = 343
LATENCY_FLOOR_US = 6.7
PUSH_BYTES_ROW = 7 * (8 + 4 * 512)
ROWS = tuple(int(x) for x in __import__("os").environ.get("ROWS", "12,24,40,65,80,128,192").split(","))


def inputs(rank, rows, dev, n=100):
    g = torch.Generator(device=dev).manual_seed(rank * 31 + rows)
    sets = []
    for _ in range(n):
        scores = torch.randn(rows, K, generator=g, device=dev) * 4
        idx = (torch.rand(rows, K, generator=g, device=dev).argsort(1).int() % (PAGE * BLOCKS)).int()
        bt = torch.randint(0, 1 << 20, (rows, BLOCKS), generator=g, device=dev, dtype=torch.int32)
        sets.append((scores, idx, bt, torch.empty(rows * K, dtype=torch.int32, device=dev), torch.empty(rows + 1, dtype=torch.int32, device=dev)))
    return sets


def time_graph(fn, sets, group, capture=contextlib.nullcontext):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for s in sets:
            fn(*s)
    torch.cuda.synchronize()
    dist.barrier(group)
    g = torch.cuda.CUDAGraph()
    with capture(), torch.cuda.graph(g):
        for s in sets:
            fn(*s)
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(group)
    ts = []
    for _ in range(15):
        total = 0.0
        for _ in range(4):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            total += a.elapsed_time(b)
        t = torch.tensor([total * 1000 / (4 * len(sets))], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item())
    del g
    return min(ts)


def run(rank, port, with_stock):
    import ctypes
    import faulthandler
    ctypes.CDLL("libc.so.6").prctl(1, 9)
    faulthandler.dump_traceback_later(200, exit=True)
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank)
    group = dist.group.WORLD
    dev = torch.device("cuda", rank)
    comm = TopkGatherMerge(rank, group, dev)
    comm.warmup()
    stock = None
    if with_stock:
        from vllm.distributed.device_communicators.aiter_custom_all_reduce import AiterCustomAllreduce
        from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import merge_dcp_scores

        ca = AiterCustomAllreduce(group, dev, max_size=1 << 25)

        def stock(scores, idx, bt, slots, indptr):
            gathered = ca.custom_all_gather(scores.view(torch.int32), dim=1).view(torch.float32)
            return merge_dcp_scores(gathered, idx, bt, rank, WORLD, PAGE, STRIDE)

        def stock_gather(scores, idx, bt, slots, indptr):
            return ca.custom_all_gather(scores.view(torch.int32), dim=1)

        def stock_merge(scores, idx, bt, slots, indptr):
            return merge_dcp_scores(scores, idx, bt, rank, WORLD, PAGE, STRIDE)

    if rank == 0:
        print(f"{'rows':>5} {'stock us':>9} {'fused us':>9} {'speedup':>8} {'AG only':>8} {'pushed MB':>10} {'BW floor':>9} {'lat floor':>10} {'% of floor':>11}", flush=True)
    for rows in ROWS:
        sets = inputs(rank, rows, dev)
        t_f = time_graph(lambda *s: comm(*s, PAGE, STRIDE), sets, group)
        if stock is not None:
            t_s = time_graph(stock, sets, group, ca.capture)
            t_g = time_graph(stock_gather, sets, group, ca.capture)
            wide = [(torch.cat([x[0]] * WORLD, dim=1), *x[1:]) for x in sets]
            t_m = time_graph(stock_merge, wide, group)
            if rank == 0:
                print(f"   stock parts: all-gather {t_g:.1f} us, merge_dcp_scores {t_m:.1f} us", flush=True)
        else:
            t_s = t_g = float("nan")
        if os.environ.get("DCP8_TIMING"):
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                for s_ in sets[:20]:
                    comm(*s_, PAGE, STRIDE)
            for _ in range(3):
                dist.barrier(group)
                g2.replay()
            torch.cuda.synchronize()
            d = comm.dbg.view(MAX_ROWS, 32)[:rows, :28].cpu()
            if rank == 0:
                d = (d - d[:, :1]).double() * 10 / 1000
                m = d.mean(0).tolist()
                names = {26: "scores loaded", 27: "cut known", 1: "keys pushed", 2: "headers received", 21: "keys received", 22: "hist done", 3: "bin picked", 23: "pass1", 24: "pass2",
                         12: "prior", 6: "thr bcast", 7: "counts done"}
                print("   phases (us from kernel start, mean over rows):", ", ".join(f"{n} {m[k]:.1f}" for k, n in names.items()),
                      "| last row end %.1f" % d[-1, 7].item(), flush=True)
        s0, i0, b0, o0, p0 = sets[0]
        comm(s0, i0, b0, o0, p0, PAGE, STRIDE)
        torch.cuda.synchronize()
        planes = [torch.empty_like(s0.cpu()) for _ in range(WORLD)]
        dist.all_gather(planes, s0.cpu(), group=group)
        want_slots, want_indptr = reference(torch.stack(planes), i0.cpu(), b0.cpu(), rank, PAGE, STRIDE)
        exact = torch.equal(p0.cpu(), want_indptr) and torch.equal(o0[: want_slots.numel()].cpu(), want_slots)
        bad = torch.tensor([0 if exact else 1])
        dist.all_reduce(bad, group=group)
        if rank == 0:
            print(f"   exact vs reference on all ranks: {bad.item() == 0}", flush=True)
        mb = PUSH_BYTES_ROW * rows / 1e6
        bw_floor = mb / XGMI_OUT_GBS * 1e3
        floor = max(bw_floor, LATENCY_FLOOR_US)
        if rank == 0:
            print(f"{rows:>5} {t_s:>9.1f} {t_f:>9.1f} {t_s / t_f:>8.2f} {t_g:>8.1f} {mb:>10.2f} {bw_floor:>9.1f} {LATENCY_FLOOR_US:>10.1f} {100 * floor / t_f:>11.0f}", flush=True)
        del sets
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(port, "--no-stock" not in sys.argv), nprocs=WORLD, join=True)
