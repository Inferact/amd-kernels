# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The warm-up compiles every row tier of the GEMM and of the gather: no compile afterwards."""
import torch
import torch.distributed as dist
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720 import pack_weight
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720.comm import ConcatLinearComm
from tests.multi_gpu import run_on_4_gpus


def worker(rank, errors):
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda', rank)
    w = torch.zeros((1536, 30720), device=dev, dtype=torch.bfloat16)
    c = ConcatLinearComm(pack_weight(w), rank, dist.group.WORLD)
    c.warm_up()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError('compile after warm-up')
    MlirCompiler.compile = unexpected
    try:
        for m in (1, 7, 8, 16, 17, 63, 64, 65, 70, 80, 81, 112, 128, 129, 224, 256, 257, 448, 512, 513, 896, 1024, 1025, 1792, 2048):
            taps = [torch.zeros((m, 6144), device=dev, dtype=torch.bfloat16) for _ in range(5)]
            out = torch.empty((m, 6144), device=dev, dtype=torch.bfloat16)
            c(taps, out)
            torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
    c.close()


def test_warmup():
    run_on_4_gpus(worker, timeout=300)
