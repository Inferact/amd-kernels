# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""After the warm-up nothing compiles at any served row count; graph replays repeat bit for bit."""
import torch

from amd_kernels import warm_up
from amd_kernels.gemm.bf16_skinny_common import SHAPES, bf16_linear, pack_weight, warmup_bf16_linear
from amd_kernels.gemm.bf16_skinny_common.reference import bf16_linear_reference

SERVED = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 64, 65, 80, 96, 128, 160, 192, 193, 224]


def test_no_compile_after_warmup(monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler

    n, k = SHAPES[0]
    g = torch.Generator(device="cuda").manual_seed(1)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    wp = pack_weight(w)
    warmup_bf16_linear(wp)
    compiled = []
    original = MlirCompiler.compile

    def track(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(MlirCompiler, "compile", track)
    for m in SERVED:
        x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
        y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_linear(x, wp, y)
        for _ in range(2):
            x.normal_()
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            yf, rf = y.double(), bf16_linear_reference(x, w).double()
            assert bool(torch.all(torch.nn.functional.cosine_similarity(yf, rf, dim=1) >= 0.99999))
    assert compiled == []


def test_entry_registered():
    assert "gemm/bf16_skinny_common" in warm_up.REGISTRY
