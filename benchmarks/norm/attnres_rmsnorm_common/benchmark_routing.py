# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the entry adds to a call on 8 ranks: the communicator comes from ``attnres_comm`` and the call is its method,
so graph replay and eager host time of the method (the entry's path) are set against the package function on the same
communicator, plus the host time of ``allreduce_attnres_rmsnorm_blocker`` (what a caller runs per call to
decide). The pre-MLP call: prefix, 8 bank rows, 3 valid, output norm. Graph: 20 calls per graph, min of 15 rounds of 5
replays, us per call; host: min over 50 batches of 100 calls of the time the Python call takes (no synchronize inside
the batch), us per call; the max over ranks.
Run: torchrun --nproc_per_node 8 benchmarks/norm/attnres_rmsnorm_common/benchmark_routing.py
"""
import importlib
import os
import time

import torch
import torch.distributed as dist

from amd_kernels.norm import attnres_rmsnorm_common as N
from amd_kernels.norm.attnres_rmsnorm_common import host as entry


def graph_us(fns, calls=20, rounds=15):
    """Per function, min over ``rounds`` of 5 replays of a graph of ``calls`` calls; the graphs alternate."""
    graphs = []
    for fn in (fns[0], *fns):  # the first graph captured replays slower whichever call it holds: discarded
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
        graph.replay()
        graphs.append(graph)
    graphs = graphs[1:]
    torch.cuda.synchronize()
    best = [float("inf")] * len(fns)
    for _ in range(rounds):
        for i, graph in enumerate(graphs):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                graph.replay()
            b.record()
            b.synchronize()
            best[i] = min(best[i], a.elapsed_time(b) * 1e3 / (5 * calls))
    return best


def host_us(fns, calls=100, rounds=50):
    """Per function, min over ``rounds`` batches of ``calls`` calls of the host time a call takes; batches alternate."""
    best = [float("inf")] * len(fns)
    for _ in range(rounds):
        for i, fn in enumerate(fns):
            dist.barrier()
            t = time.perf_counter()
            for _ in range(calls):
                fn()
            best[i] = min(best[i], (time.perf_counter() - t) / calls * 1e6)
            torch.cuda.synchronize()
    return best


def worst(v):
    t = torch.tensor([v], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    dev = torch.device("cuda", rank)
    host = importlib.import_module(entry._ROUTES.hosts[(7168, 8)])
    comm = N.attnres_comm(rank, dist.group.WORLD, dev, hidden=7168)
    if rank == 0:
        print("| M | graph package | graph method | host package | host method | host reason |")
        print("|---|---|---|---|---|---|")
    for m in (1, 64, 256):
        bf = torch.bfloat16
        x = torch.randn(m, 7168, device=dev).to(bf)
        prefix = torch.randn(m, 7168, device=dev).to(bf)
        bank = torch.randn(m, 8, 7168, device=dev).to(bf)
        nw, qk, ow = (torch.ones(7168, dtype=bf, device=dev) for _ in range(3))
        args = (x, prefix, bank, nw, qk, 3, -1, ow)
        package = lambda: host.allreduce_attnres_rmsnorm(comm, *args)
        method = lambda: comm.allreduce_attnres_rmsnorm(*args)
        g = [worst(v) for v in graph_us((package, method))]
        h = [worst(v) for v in host_us((package, method))]
        reason = worst(host_us((lambda: N.allreduce_attnres_rmsnorm_blocker(comm, *args),))[0])
        if rank == 0:
            print(f"| {m} | {g[0]:.2f} | {g[1]:.2f} | {h[0]:.2f} | {h[1]:.2f} | {reason:.2f} |", flush=True)
    torch.cuda.synchronize()
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
