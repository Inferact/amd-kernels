# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The first call of each mode compiles every size; no served size compiles afterwards."""
import torch
import torch.distributed as dist

from .harness import run_on_8_gpus


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.collectives.handshake_ranks8 import barrier, exchange, make_comm
    from amd_kernels.collectives.handshake_ranks8.host import SIZES
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device("cuda", rank)
    warm_up.set_auto(True)
    comm = make_comm(rank, dist.group.WORLD, dev)
    compiles = []
    real = MlirCompiler.compile

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    def call(n):
        x = torch.zeros((8, n), dtype=torch.int32, device=dev)
        exchange(comm, x, torch.empty_like(x))
        torch.cuda.synchronize()

    barrier(comm)
    call(64)
    MlirCompiler.compile = counting
    try:
        barrier(comm)
        for n in SIZES:
            call(n)
    finally:
        MlirCompiler.compile = real
    if compiles:
        errors.append(f"rank {rank}: {len(compiles)} compiles after the warm-up")


def test_no_compile_after_warm_up():
    run_on_8_gpus(_worker, timeout=900)
