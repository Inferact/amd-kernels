# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The first call of each mode compiles every token tier; no served token count compiles afterwards."""
import torch
import torch.distributed as dist

from .harness import run_on_8_gpus

PROBES = (1, 12, 16, 17, 24, 33, 40, 48, 65, 80, 96, 128, 129, 160, 192)


def _worker(rank, errors):
    from amd_kernels import warm_up
    from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import DcpVupCombine, dcp8_lse_combine_vup, prepare_w_uv
    from flydsl.compiler.jit_function import MlirCompiler

    dev = torch.device("cuda", rank)
    warm_up.set_auto(True)
    comm = DcpVupCombine(rank, dist.group.WORLD, dev)
    wt = prepare_w_uv(torch.zeros((96, 512, 128), dtype=torch.bfloat16, device=dev))
    compiles = []
    real = MlirCompiler.compile

    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)

    def call(m, gate):
        x = torch.zeros((m, 96, 512), dtype=torch.bfloat16, device=dev)
        lse = torch.zeros((m, 96), dtype=torch.float32, device=dev)
        out = torch.empty((m, 12, 128), dtype=torch.bfloat16, device=dev)
        g = torch.zeros((m, 1536), dtype=torch.bfloat16, device=dev) if gate else None
        dcp8_lse_combine_vup(comm, x, lse, wt, out, gate=g)
        torch.cuda.synchronize()

    for gate in (False, True):
        call(12, gate)
        MlirCompiler.compile = counting
        try:
            for m in PROBES:
                call(m, gate)
        finally:
            MlirCompiler.compile = real
        if compiles:
            errors.append(f"rank {rank} gate={gate}: {len(compiles)} compiles after the warm-up")
        compiles.clear()


def test_no_compile_after_warm_up():
    run_on_8_gpus(_worker)
