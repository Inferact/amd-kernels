# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: the pull grid (remote, local workgroups) of ``RowAllToAll`` per owned-row count; stage + pull, graph
replay, MAX over ranks, median us and the link rate, the five best grids per row count (``host.PULL``).

    python3 benchmarks/collectives/all_to_all_bf16_n100032_ranks2/sweep_pull.py [owned rows ...]
"""
import socket
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from amd_kernels.collectives.all_to_all_bf16_n100032_ranks2 import RowAllToAll, kernel
from amd_kernels.collectives.all_to_all_bf16_n100032_ranks2.reference import N, WORLD

GRIDS = [(4, 2), (8, 2), (8, 4), (12, 4), (16, 2), (16, 4), (16, 8), (24, 4), (24, 8), (32, 8), (48, 8), (64, 8),
         (64, 16), (96, 16), (128, 16), (128, 32), (192, 32)]


def run(rank, port, owned):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank)
    comm = RowAllToAll(rank, dist.group.WORLD, dev, max_rows=2048)
    comm.warmup()
    pulls = {gr: kernel.compile_pull(*gr) for gr in GRIDS}
    gen = torch.Generator(device=dev).manual_seed(rank)
    for own in owned:
        m = own * WORLD
        cs = torch.tensor([own] * WORLD, dtype=torch.int32, device=dev)
        nsets = max(4, min(50, 600_000_000 // (m * N * 2)))
        xs = [torch.randint(-32768, 32768, (m, N), device=dev, dtype=torch.int16, generator=gen).view(torch.bfloat16)
              for _ in range(nsets)]
        out = torch.empty(own, WORLD * N, dtype=torch.bfloat16, device=dev)
        res = []
        for gr in GRIDS:
            comm.pull_exe = pulls[gr]
            for x in xs[:2]:
                comm(x, cs, out)
            torch.cuda.synchronize()
            dist.barrier()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for x in xs:
                    comm(x, cs, out)
            g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                dist.barrier()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(5):
                    g.replay()
                b.record()
                b.synchronize()
                t = torch.tensor([a.elapsed_time(b) * 1000 / (5 * nsets)])
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                ts.append(t.item())
            res.append((statistics.median(ts), gr))
            del g
        res.sort()
        if rank == 0:
            print(f"own {own}: " + " | ".join(f"{gr} {t:.1f} ({own * N * 2 / t / 1e3:.1f} GB/s)" for t, gr in res[:5]),
                  flush=True)
        del xs, out
        torch.cuda.empty_cache()
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    owned = [int(v) for v in sys.argv[1:]] or [4, 8, 16, 24, 32, 48, 64, 128, 256, 512]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(run, args=(port, owned), nprocs=WORLD, join=True)
