# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight ranks, CUDA-graph replay, cold rotated weights, MAX over ranks, min / median of the rounds (us per call).

Stock replicated: vLLM ``rocm_unquantized_gemm_impl`` of [R, 35840] x [7168, 35840]^T (514 MB weight) + ``rms_norm``.
Stock sharded: the same GEMM on the 896-row shard (64 MB) + AITER ``custom_all_gather`` + ``rms_norm``.
This entry: ``context_proj_gather_norm`` on the packed activations (``pack_aux`` is the eager copy that replaces the
``torch.cat`` into the staging buffer and is reported apart).
Run on eight otherwise idle GPUs: python3 benchmarks/gemm/bf16_linear_gather_n896_k35840_ranks8/benchmark_context_proj.py
"""
import argparse
import os
import statistics
import sys

import torch
import torch.distributed as dist

CONFS = {"c16": 40, "c24": 65, "c32": 80, "c48": 192}
CEIL_US = 11.34  # 64 MB cold HBM read, measured: 5.9 TB/s incl. launch


def timeit(ca, fns, n, rounds, reps):
    graphs = []
    for fn in fns:
        for i in range(2):
            fn(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with ca.capture(), torch.cuda.graph(g):
            for i in range(n):
                fn(i)
        graphs.append(g)
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    ts = [[] for _ in graphs]
    for r in range(rounds):
        for k in (range(len(graphs)) if r % 2 == 0 else range(len(graphs) - 1, -1, -1)):
            dist.barrier()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(reps):
                graphs[k].replay()
            b.record()
            b.synchronize()
            ts[k].append(a.elapsed_time(b) * 1e3 / (reps * n))
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, ts)
    return [[max(v[k][r] for v in every) for r in range(rounds)] for k in range(len(graphs))]


def worker(rank, world, port, rows_list, rounds):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    import vllm._custom_ops as ops
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl as gemm

    from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8 import ProjComm, context_proj_gather_norm, host
    from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.reference import K, N_FULL, N_SHARD

    dev = torch.device("cuda", rank)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    comm = ProjComm(rank, dist.group.WORLD, dev)
    nc = 6
    g = torch.Generator(device="cuda").manual_seed(rank)
    shards = [(torch.randn(N_SHARD, K, device=dev, generator=g) * 0.02).bfloat16() for _ in range(nc)]
    packed = [host.pack_weight(w) for w in shards]
    full = [(torch.randn(N_FULL, K, device=dev, generator=g) * 0.02).bfloat16() for _ in range(2)]
    nw = (torch.rand(N_FULL, device=dev, generator=g) + 0.5).bfloat16()
    n = 24
    table = []
    for rows in rows_list:
        aux = [(torch.randn(rows, K // 5, device=dev, generator=g) * 0.4).bfloat16() for _ in range(5)]
        x = torch.cat(aux, dim=-1)
        xp = host.packed_rows_buffer(rows, dev)
        host.pack_aux(aux, xp, rows)
        out = torch.empty(rows, N_FULL, dtype=torch.bfloat16, device=dev)

        def replicated(i):
            y = gemm(x, full[i % 2])
            o = torch.empty_like(y)
            ops.rms_norm(o, y, nw, 1e-5)
            return o

        def sharded(i):
            y = ca.custom_all_gather(gemm(x, shards[i % nc]), dim=-1)
            o = torch.empty_like(y)
            ops.rms_norm(o, y, nw, 1e-5)
            return o

        def new(i):
            return context_proj_gather_norm(comm, xp, packed[i % nc], nw, rows, out=out)

        def pack(i):
            return host.pack_aux(aux, xp, rows)

        if os.environ.get('BENCH_DEBUG'):
            for name_, f_ in (('replicated', replicated), ('sharded', sharded), ('new', new), ('pack', pack)):
                f_(0)
                torch.cuda.synchronize()
                if rank == 0:
                    print('eager ok', name_, rows, flush=True)
        res = timeit(ca, [replicated, sharded, new, pack], n, rounds, 10)
        table.append((rows, [(min(r), statistics.median(r)) for r in res]))
    if rank == 0:
        print("rows | stock replicated | stock sharded+AG | new (min / median us) | pack_aux | new vs sharded | % of 64 MB ceiling",
              flush=True)
        for rows, r in table:
            (a, am), (b, bm), (c, cm), (d, dm) = r
            print(f"{rows:4d} | {a:7.1f} / {am:7.1f} | {b:6.1f} / {bm:6.1f} | {c:6.1f} / {cm:6.1f} | {d:5.1f} | "
                  f"{b / c:4.2f}x | {CEIL_US / c * 100:3.0f}%", flush=True)
    dist.barrier()
    comm.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=list(CONFS.values()))
    p.add_argument("--rounds", type=int, default=15)
    a = p.parse_args()
    port = 20000 + os.getpid() % 20000
    torch.multiprocessing.spawn(worker, args=(8, port, a.rows, a.rounds), nprocs=8)
