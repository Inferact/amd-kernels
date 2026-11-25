# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the DCP8 combine of packed partials against vLLM's a2a merge (RCCL ``all_to_all_single`` of the
send buffer + Triton ``_dcp_a2a_unpack_combine``) on 8 GPUs: inputs rotating over > 300 MB, MAX over ranks, median
and min of 9 replays, us per call, bit equality of the two outputs, and a table against the ceiling model.

    python3 benchmarks/attention/dcp8_lse_combine_packed_h16_v192/benchmark_combine.py [--tokens 12,24,...]

Ceiling model (the constants of attention/dcp8_lse_combine_vup_mla_h12_kv512, measured on MI355X): launch 1.6 us +
flag one-way 0.8 us + bytes sent to the 7 peers / 343 GB/s all-to-all egress + combine tail 0.6 us.
"""
import argparse
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
CALLS = 16
ROUNDS = 9
ROW = 48 * 66 * 2
EGRESS = 343e9


def ceiling_us(m):
    return 1.6 + 0.8 + 7 * m * ROW / EGRESS * 1e6 + 0.6


def _worker(rank, port, tokens, out_q):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192 import DcpPackedCombine, dcp8_lse_combine_packed
    from vllm.v1.attention.ops.dcp import _dcp_a2a_unpack_combine_kernel

    dev = torch.device("cuda", rank)
    nccl = dist.new_group(backend="nccl")
    comm = DcpPackedCombine(rank, dist.group.WORLD, dev)
    g = torch.Generator(device=dev).manual_seed(rank)
    rows_out = []
    for m in tokens:
        nrot = max(4, min(CALLS, -(-300 * 2**20 // (WORLD * m * ROW))))
        sends = []
        for _ in range(nrot):
            s = torch.empty((WORLD, m, 48, 66), dtype=torch.bfloat16, device=dev)
            s[..., :64] = (torch.randn((WORLD, m, 48, 64), generator=g, device=dev) * 0.5).bfloat16()
            lse = torch.randn((WORLD, m, 48), generator=g, device=dev) * 3 + 10
            s[..., 64:] = lse.unsqueeze(-1).view(torch.bfloat16)
            sends.append(s)
        outs = [torch.empty((m, 16, 192), dtype=torch.bfloat16, device=dev) for _ in range(nrot)]
        recv = torch.empty((WORLD, m, 48, 66), dtype=torch.bfloat16, device=dev)

        def ours(i):
            dcp8_lse_combine_packed(comm, sends[i], outs[i])

        def stock(i):
            dist.all_to_all_single(recv, sends[i], group=nccl)
            o = outs[i].view(m, 48, 64)
            _dcp_a2a_unpack_combine_kernel[(m, 48)](
                recv, o, o, *recv.stride(), *o.stride(), 0, 0,
                N=WORLD, HEAD_DIM=64, IS_BASE_E=False, RETURN_LSE=False, LSE_PACK_DIM=2,
            )

        ours(0)
        torch.cuda.synchronize()
        mine = outs[0].clone()
        stock(0)
        torch.cuda.synchronize()
        same = torch.equal(mine.view(torch.int16), outs[0].view(torch.int16))

        graphs = {}
        for name, fn in (("ours", ours), ("stock", stock)):
            for i in range(nrot):
                fn(i)
            torch.cuda.synchronize()
            dist.barrier()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                for c in range(CALLS):
                    fn(c % nrot)
            graphs[name] = gr
        times = {k: [] for k in graphs}
        for rd in range(ROUNDS):
            for name in (("ours", "stock") if rd % 2 == 0 else ("stock", "ours")):
                graphs[name].replay()
                torch.cuda.synchronize()
                dist.barrier()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                graphs[name].replay()
                e.record()
                torch.cuda.synchronize()
                t = torch.tensor([s.elapsed_time(e) * 1000 / CALLS])
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                times[name].append(t.item())
        ok = torch.tensor([int(same)])
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        rows_out.append((m, {k: (statistics.median(v), min(v)) for k, v in times.items()}, bool(ok.item())))
        del graphs
        torch.cuda.empty_cache()
    if rank == 0:
        out_q.put(rows_out)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,12,32,64,128,192,240,288,384,512")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, tokens, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    rows = q.get(timeout=1800)
    for p in procs:
        p.join()
    print(f"\nDCP8 combine of packed partials: graph replay, us per call (median/min of {ROUNDS}), MAX over 8 ranks")
    print(f"{'M':>4} {'sent MB':>8} {'ceiling':>8} {'RCCL a2a + Triton':>17} {'fused':>13} {'speedup':>8} {'% ceiling':>9} {'bit-equal':>9}")
    for m, t, same in rows:
        c = ceiling_us(m)
        print(f"{m:>4} {7 * m * ROW / 1e6:>8.2f} {c:>8.1f} {t['stock'][0]:>11.1f}/{t['stock'][1]:<5.1f} "
              f"{t['ours'][0]:>7.1f}/{t['ours'][1]:<5.1f} {t['stock'][0] / t['ours'][0]:>7.2f}x {100 * c / t['ours'][1]:>8.0f}% "
              f"{'yes' if same else 'NO':>9}")


if __name__ == "__main__":
    main()
