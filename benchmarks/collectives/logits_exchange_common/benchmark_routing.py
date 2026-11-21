# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the logits exchange entry adds over the package: ``gemm_exchange`` on a communicator built by the entry vs one
built by the package (the entry adds no call of its own: dispatch 0), CUDA-graph replay, the host time of an eager
call (enqueue only) and of the call's reason; all-gather mode, min over the rounds of the per-round mean, max over the
eight ranks, us per call.

Run on eight GPUs: python3 benchmarks/collectives/logits_exchange_common/benchmark_routing.py
"""
import argparse
import importlib
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N_GRAPH, REPS, N_EAGER = 24, 10, 200


def _sync():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _worst(v):
    if not dist.is_initialized():
        return v
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, v)
    return [max(e[i] for e in every) for i in range(len(v))]


def timeit(fns, rounds):
    """[(graph us, eager us)] per fn."""
    graphs = []
    for fn in fns:
        fn()
        _sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(N_GRAPH):
                fn()
        graphs.append(g)
    gt, et = [[] for _ in fns], [[] for _ in fns]
    for r in range(rounds):
        for i in range(len(fns)) if r % 2 == 0 else range(len(fns) - 1, -1, -1):
            _sync()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(REPS):
                graphs[i].replay()
            b.record()
            b.synchronize()
            gt[i].append(a.elapsed_time(b) * 1e3 / (REPS * N_GRAPH))
            _sync()
            t = time.perf_counter()
            for _ in range(N_EAGER):
                fns[i]()
            et[i].append((time.perf_counter() - t) * 1e6 / N_EAGER)
            torch.cuda.synchronize()
    return [(min(_worst(g)), min(_worst(e))) for g, e in zip(gt, et)]


def host_us(fn, rounds):
    """Host us per call of a host-only function (the reason)."""
    ts = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(N_EAGER):
            fn()
        ts.append((time.perf_counter() - t) * 1e6 / N_EAGER)
    return min(_worst([min(ts)]))


def _print(rows, res, dispatch, reason):
    (ge, ee), (gd, ed) = res
    print(f"| {rows} | {gd:.2f} | {ge:.2f} | {ed:.2f} | {ee:.2f} | {dispatch:.2f} | {reason:.2f} |", flush=True)


HEADER = ("| rows | graph direct | graph entry | host direct | host entry | dispatch | reason |\n"
          "|---|---|---|---|---|---|---|")


def _rank(rank, port, rows_list, rounds):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=8)
    from amd_kernels.collectives.logits_exchange_common import (logits_exchange_comm, logits_gemm_exchange_shapes,
                                                                logits_gemm_exchange_unsupported_reason)
    from amd_kernels.collectives.logits_exchange_common import host as entry

    dev = torch.device("cuda", rank)
    for key in sorted(logits_gemm_exchange_shapes()):
        n, k, world = key
        if world != 8:
            continue
        host = importlib.import_module(entry._SHAPE_HOSTS[key])
        comm = logits_exchange_comm(rank, dist.group.WORLD, dev, n_shard=n, k=k, max_rows=host.RECV_ROWS)
        direct = host.RowAllToAll(rank, dist.group.WORLD, dev, max_rows=host.RECV_ROWS)
        direct._local = torch.empty(host.RECV_ROWS, n, dtype=torch.bfloat16, device=dev)
        w = (torch.randn(n, k, device=dev) * 0.02).bfloat16()
        if rank == 0:
            print(f"logits gemm exchange {key}\n{HEADER}")
        for rows in rows_list:
            x = torch.randn(rows, k, device=dev).bfloat16()
            idx = torch.arange(rows, device=dev)
            res = timeit([lambda: comm.gemm_exchange(x, w, idx, None, 0),
                          lambda: direct.gemm_exchange(x, w, idx, None, 0)], rounds)
            reason = host_us(lambda: logits_gemm_exchange_unsupported_reason(comm, x, w, idx, None, 0), rounds)
            if rank == 0:
                _print(rows, res, 0.0, reason)
        comm.close()
        direct.close()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[8, 64, 256])
    p.add_argument("--rounds", type=int, default=15)
    a = p.parse_args()
    mp.spawn(_rank, args=(29500 + os.getpid() % 300, a.rows, a.rounds), nprocs=8)


if __name__ == "__main__":
    main()
