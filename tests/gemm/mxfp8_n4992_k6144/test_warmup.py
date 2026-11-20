# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""After ``warmup_mxfp8_gemm_weight`` of the N 4992 weight for 32768 tokens nothing compiles: every entry at every
window edge and at random row counts, the deferred partials, and the DBO plain grid."""

import random

import torch

from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_n4992_k6144 import host as H

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950
from .test_n4992_k6144 import DECODE_MS, PERSISTENT_MS

N, K = H.N, H.K
MAX_TOKENS = 32768


@torch.inference_mode()
def test_nothing_compiles_after_warmup(monkeypatch):
    import flydsl.compiler as flyc
    from flydsl.compiler.jit_function import MlirCompiler

    require_gfx950()
    _, _, _, (w, ws) = make_inputs(16, N, K)
    g.warmup_mxfp8_gemm_weight(w, ws, MAX_TOKENS, torch.bfloat16)
    rng = random.Random(4992)
    ms = sorted(set(DECODE_MS + PERSISTENT_MS + [rng.randint(1, MAX_TOKENS) for _ in range(24)] + [MAX_TOKENS]))
    x = torch.zeros((MAX_TOKENS, K), dtype=torch.bfloat16, device="cuda")
    compiled = []
    real = MlirCompiler.compile
    real_jit = flyc.compile
    # a variant's first launch in a process goes through flyc.compile even when the disk cache has it
    monkeypatch.setattr(MlirCompiler, "compile", lambda self, *a, **kw: compiled.append(1) or real(self, *a, **kw))
    monkeypatch.setattr(flyc, "compile", lambda *a, **kw: compiled.append(2) or real_jit(*a, **kw))
    for dbo in (False, True):
        monkeypatch.setattr(g, "DBO", dbo)
        for m in ms:
            tile = g.prequant_code_tile_rows(m, N, K)
            a, asc = g.mxfp8_e4m3_quantize_shuffled(x[:m], tile) if tile else g.mxfp8_e4m3_quantize_aiter(x[:m])
            g.mxfp8_gemm_prequant(a, asc, w, ws, torch.bfloat16, m)
            if tile:
                g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m, defer_reduce=True)
            g.mxfp8_linear(x[:m], w, ws)
            assert not compiled, (dbo, m)
    torch.cuda.synchronize()
