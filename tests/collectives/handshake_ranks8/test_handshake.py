# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract against reference.py on 8 GPUs: every size and path eager, then thousands of back-to-back hand-offs in CUDA
graphs with a payload check of every call, mixed with other kernels (L2 thrash, GEMM, random per-rank delays), changed
inputs per replay, and the epoch counter crossing the int32 wrap."""
import random

import torch
import torch.distributed as dist

from .harness import run_on_8_gpus

W = 8


def _eager(rank, errors):
    from amd_kernels.collectives.handshake_ranks8 import barrier, exchange, make_comm
    from amd_kernels.collectives.handshake_ranks8.host import FLAG_SIZES, LL_SIZES
    from amd_kernels.collectives.handshake_ranks8.reference import exchange_reference, payload

    dev = torch.device("cuda", rank)
    comm = make_comm(rank, dist.group.WORLD, dev)
    for variant in ("wave", "lane", "pull"):
        barrier(comm, variant)
    for path, sizes in (("ll", LL_SIZES), ("flag", LL_SIZES + FLAG_SIZES), ("pull", LL_SIZES + FLAG_SIZES)):
        for n in sizes:
            for call in range(6):
                xs = [payload(0, call, s, n, dev) for s in range(W)]
                out = torch.full((W, n), -1, dtype=torch.int32, device=dev)
                exchange(comm, xs[rank], out, path)
                torch.cuda.synchronize()
                want = exchange_reference(xs, rank)
                if not torch.equal(out, want):
                    errors.append(f"rank {rank} {path} n={n} call {call}: {int((out != want).sum())} words differ")
    dist.barrier()


def test_eager_sizes_and_paths():
    run_on_8_gpus(_eager)


def _graph(rank, errors, replays, calls, wrap):
    from amd_kernels.collectives.handshake_ranks8 import barrier, exchange, make_comm
    from amd_kernels.collectives.handshake_ranks8.reference import exchange_reference, payload

    dev = torch.device("cuda", rank)
    comm = make_comm(rank, dist.group.WORLD, dev)
    if wrap:
        comm.ctr.fill_(2**31 - 1 - calls)
        torch.cuda.synchronize()
        dist.barrier()
    plan = [("ll", 64, 1), ("flag", 1024, 1), ("barrier", 0, 1), ("pull", 2048, 1), ("ll", 512, 1), ("ll", 4096, 8),
            ("flag", 4096, 4), ("ll", 256, 1), ("flag", 16384, 16), ("ll", 16384, 32), ("ll", 128, 1)]
    plan = [plan[(i * 5) % len(plan)] for i in range(calls)]
    xs = [torch.zeros((W, n), dtype=torch.int32, device=dev) for _, n, _ in plan]
    outs = [torch.zeros((W, n), dtype=torch.int32, device=dev) for _, n, _ in plan]
    thrash = torch.empty(32 << 20, dtype=torch.float32, device=dev)
    thrash2 = torch.empty_like(thrash)
    a = torch.randn(512, 512, device=dev, dtype=torch.bfloat16)
    rng = random.Random(rank)
    delays = [rng.randrange(40000) if i % 3 == 0 else 0 for i in range(calls)]

    def body():
        for i, (path, n, nc) in enumerate(plan):
            if delays[i]:
                torch.cuda._sleep(delays[i])
            if i % 4 == 1:
                thrash2.copy_(thrash)
            if i % 4 == 3:
                a.mm(a)
            if path == "barrier":
                barrier(comm, ("wave", "lane", "pull")[i % 3])
            else:
                exchange(comm, xs[i], outs[i], path, nc)

    body()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        body()
    for rp in range(replays):
        for i, (path, n, _) in enumerate(plan):
            if path != "barrier":
                xs[i].copy_(payload(rp, i, rank, n, dev))
                outs[i].fill_(-1)
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        for i, (path, n, _) in enumerate(plan):
            if path == "barrier":
                continue
            want = exchange_reference([payload(rp, i, s, n, dev) for s in range(W)], rank)
            if not torch.equal(outs[i], want) and len(errors) < 4:
                bad = int((outs[i] != want).sum())
                errors.append(f"rank {rank} replay {rp} call {i} {path} n={n}: {bad} words differ")
    dist.barrier()


def test_graph_stress():
    run_on_8_gpus(_graph, 200, 64, False, timeout=110)


def test_graph_epoch_wrap():
    run_on_8_gpus(_graph, 6, 64, True, timeout=110)
