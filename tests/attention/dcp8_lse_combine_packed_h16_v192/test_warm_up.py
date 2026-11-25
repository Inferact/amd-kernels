# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The first call compiles the single variant (the row count is a runtime argument); no row count compiles after."""
import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus

PROBES = (1, 12, 16, 17, 33, 100, 128, 129, 255, 288, 512)


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.attention.dcp8_lse_combine_packed_h16_v192 import DcpPackedCombine, dcp8_lse_combine_packed
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device("cuda", rank)
    warm_up.set_auto(True)
    comm = DcpPackedCombine(rank, dist.group.WORLD, dev)
    compiles = []
    real = MlirCompiler.compile

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    def call(m):
        send = torch.zeros((8, m, 48, 66), dtype=torch.bfloat16, device=dev)
        out = torch.empty((m, 16, 192), dtype=torch.bfloat16, device=dev)
        dcp8_lse_combine_packed(comm, send, out)
        torch.cuda.synchronize()

    call(12)
    MlirCompiler.compile = counting
    try:
        for m in PROBES:
            call(m)
    finally:
        MlirCompiler.compile = real
    if compiles:
        errors.append(f"rank {rank}: {len(compiles)} compiles after the warm-up")


def test_no_compile_after_warm_up():
    run_on_8_gpus(_worker)
