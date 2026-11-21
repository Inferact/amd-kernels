# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight GPUs, CUDA graph replay of 16 calls over rotated weights (q_b 7.1 MB + W_UK 12.6 MB per call, more than the
last-level cache), us per call: the minimum over rounds (5 replays each) of the slowest rank. Columns: the unfused vLLM
sequence (skinny / hipBLASLt q_b GEMM, ``torch.bmm`` with W_UK, ``torch.cat``, AITER IPC all-gather); this entry's bf16
output (``q``), its fp8 hi/lo record alone (``q8``) and both (``both``); ``gather``: the same call with the consumers only
waiting for the data (no W_UK, no output); ``P0``: the push + flag protocol alone (producers push without loads or math,
consumers wait). ``floor`` is the chain model of the push all-gather from the ceilings measured on MI355X:
launch gap, q_b weight load, the push at the 8-GPU all-to-all egress rate, remote write ack, flag, the last tile's
consumer chain."""

import argparse
import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
COPIES = 16
LAUNCH_US = 1.57
HBM_READ_GBPS = 6400.0
HBM_WRITE_GBPS = 6000.0
XGMI_GBPS = 343.0
ACK_US = 0.7
FLAG_US = 0.7
LOAD_LATENCY_US = 0.35
CONSUMER_US = 1.3
ROWS = (12, 24, 40, 65, 80, 96, 128, 160, 192)


def floor_q(m):
    """Chain floor (us) of the push all-gather: launch, first weights (load latency + the 7.08 MB q_b stream), the push of
    7 x 4608 B per row at the all-to-all egress rate, ack of the last store, flag, and the last tile's consumer chain
    (two load latencies, MFMA, the last stores); the weight streams, the earlier pushes and tiles overlap."""
    qb_us = 7.08e6 / (HBM_READ_GBPS * 1e3)
    push_us = 7 * m * 4608 / (XGMI_GBPS * 1e3)
    return LAUNCH_US + LOAD_LATENCY_US + qb_us + push_us + ACK_US + FLAG_US + CONSUMER_US


def worker(rank, port, rows, rounds, queue):
    try:
        run(rank, port, rows, rounds, queue)
    except Exception:  # noqa: BLE001
        queue.put((rank, traceback.format_exc()))
        queue.close()
        queue.join_thread()
        os._exit(1)


def run(rank, port, rows, rounds, queue):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512 import (
        QueryGatherComm,
        pack_q_b,
        pack_w_uk,
        q_b_allgather_wuk_all_heads,
    )
    from amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512 import host
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(rank)
    rnd = lambda *s, scale: (torch.randn(*s, device=dev, generator=g) * scale).bfloat16()  # noqa: E731
    w_qb = [rnd(2304, 1536, scale=0.03) for _ in range(COPIES)]
    w_uk = [rnd(12, 128, 512, scale=0.05) for _ in range(COPIES)]
    wq = [pack_q_b(w) for w in w_qb]
    wu96 = [pack_w_uk(w.repeat(8, 1, 1)) for w in w_uk]
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    comm = QueryGatherComm(rank, dist.group.WORLD, dev, "q")
    result = {}
    for m in rows:
        x = torch.randn((m, 1536), device=dev, generator=g).bfloat16()

        def stock(i):
            q = rocm_unquantized_gemm_impl(x, w_qb[i]).view(m, 12, 192)
            qn = q[..., :128].transpose(0, 1)
            lat = qn.new_empty((m, 12, 512))
            torch.bmm(qn, w_uk[i], out=lat.transpose(0, 1))
            cat = torch.cat((lat, q[..., 128:]), dim=-1)
            return ca.custom_all_gather(cat.view(-1).view(torch.int32).view(m, -1), dim=1)

        out = torch.empty((m, 96, 576), dtype=torch.bfloat16, device=dev)
        q8 = torch.empty((m, 96, 1168), dtype=torch.uint8, device=dev)
        cfg = host.config_q(m)
        fns = {
            "stock": stock,
            "q": lambda i: q_b_allgather_wuk_all_heads(comm, x, wq[i], wu96[i], out=out),
            "q8": lambda i: q_b_allgather_wuk_all_heads(comm, x, wq[i], wu96[i], out_q8=q8),
            "both": lambda i: q_b_allgather_wuk_all_heads(comm, x, wq[i], wu96[i], out=out, out_q8=q8),
            "gather": lambda i: comm._push(x, wq[i], wu96[i], None, None, cfg, 3),
            "P0": lambda i: comm._push(x, wq[i], wu96[i], None, None, cfg, 3, skeleton=7),
        }
        times = {}
        for name, fn in fns.items():
            fn(0)
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(graph):
                for i in range(COPIES):
                    fn(i)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(rounds):
                dist.barrier()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record()
                for _ in range(5):
                    graph.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1e3 / (5 * COPIES))
            times[name] = ts
        result[m] = times
    comm.close()
    ca.close()
    queue.put((rank, result))
    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    p.add_argument("--rounds", type=int, default=15)
    a = p.parse_args()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, port, a.rows, a.rounds, queue)) for r in range(WORLD)]
    for pr in procs:
        pr.start()
    got = {}
    while len(got) < WORLD:
        rank, res = queue.get()
        if isinstance(res, str):
            for pr in procs:
                pr.kill()
            raise RuntimeError(f"rank {rank}:\n{res}")
        got[rank] = res
    for pr in procs:
        pr.join()
    names = ("stock", "q", "q8", "both", "gather", "P0")
    print(f"{'M':>4} " + " ".join(f"{n:>7}" for n in names) + f" {'q/stock':>8} {'floor':>6} {'%floor':>7} {'%P0':>5}")
    for m in a.rows:
        med = {k: min(max(got[r][m][k][i] for r in range(WORLD)) for i in range(a.rounds)) for k in names}
        f = floor_q(m)
        print(
            f"{m:>4} "
            + " ".join(f"{med[n]:>7.1f}" for n in names)
            + f" {med['stock'] / med['q']:>7.2f}x {f:>6.1f} {100 * f / med['q']:>6.0f}% "
            f"{100 * med['P0'] / med['q']:>4.0f}%"
        )


if __name__ == "__main__":
    main()
