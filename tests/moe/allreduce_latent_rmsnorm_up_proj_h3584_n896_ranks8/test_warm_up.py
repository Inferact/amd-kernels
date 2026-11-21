# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 on 8 ranks: the entry's first call compiles every row-count variant in compile-only mode (no launch inside
the warm-up), then every row count 1..256 compiles nothing."""
import torch
import torch.distributed as dist

from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.multi_gpu8 import run_on_8_gpus
from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.workload import K, N, rand

WALK_ROWS = (1, 7, 16, 17, 33, 48, 65, 100, 129, 191, 192, 193, 256)
SLOT_WALK_ROWS = (1, 7, 16, 17, 33, 48, 64)


def worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import (
        LatentTailComm,
        allreduce_latent_rmsnorm_up_proj,
        allreduce_slots_rmsnorm_up_proj,
    )
    from flydsl.compiler.jit_function import CompiledFunction, MlirCompiler

    warm_up.set_auto(True)  # tests/__init__.py turns it off for the single-size tests
    warm_up.set_workers(2)  # 8 ranks share the node's CPUs
    dev = torch.device("cuda", rank)
    launches = []
    launch = CompiledFunction.__call__

    def spy(self, *args):
        if warm_up._in_warm_up.get():
            launches.append(1)
        return launch(self, *args)

    CompiledFunction.__call__ = spy
    try:
        comm = LatentTailComm(rank, dist.group.WORLD, dev)
        nw, w = rand((K,), 1, dev, 0.2), rand((N, K), 2 + rank, dev, 0.03)

        def call(m, seed):
            shared = rand((m, 7168), seed + rank, dev)
            view = shared[:, rank * N:(rank + 1) * N]
            allreduce_latent_rmsnorm_up_proj(comm, rand((m, K), seed + 50 + rank, dev), nw, w, view)
            torch.cuda.synchronize()
            return view

        def call_slots(m, seed):
            shared = rand((m, 7168), seed + rank, dev)
            view = shared[:, rank * N:(rank + 1) * N]
            rs = torch.randperm(m * 16 + 5, generator=torch.Generator().manual_seed(seed))[: m * 16].to(dev, torch.int32)
            allreduce_slots_rmsnorm_up_proj(comm, rand((m * 16 + 5, K), seed + 70 + rank, dev), rs, nw, w, view)
            torch.cuda.synchronize()
            return view

        call(12, 7)
        call_slots(12, 8)
    finally:
        CompiledFunction.__call__ = launch
    if launches:
        errors.append(f"rank {rank}: {len(launches)} launches inside a warm-up")
    late0 = warm_up.late_compiles()
    compiles = []
    real = MlirCompiler.compile

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    MlirCompiler.compile = counting
    try:
        for m in WALK_ROWS:
            out = call(m, 100 + m)
            if compiles:
                errors.append(f"rank {rank}: {len(compiles)} compiles at {m} rows after the warm-up")
                compiles.clear()
            if not bool(torch.isfinite(out).all()):
                errors.append(f"rank {rank}: non-finite output at {m} rows")
        for m in SLOT_WALK_ROWS:
            out = call_slots(m, 200 + m)
            if compiles:
                errors.append(f"rank {rank}: {len(compiles)} compiles at {m} slot rows after the warm-up")
                compiles.clear()
            if not bool(torch.isfinite(out).all()):
                errors.append(f"rank {rank}: non-finite slot output at {m} rows")
    finally:
        MlirCompiler.compile = real
    if warm_up.late_compiles() != late0:
        errors.append(f"rank {rank}: {warm_up.late_compiles() - late0} late compiles under an entry")
    comm.close()


def test_warm_up_walk():
    run_on_8_gpus(worker, timeout=1200)
