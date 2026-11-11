# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entries a model integration calls instead of choosing layouts itself: ``preshuffle_mxfp8_weight_`` (in
place, refuses unexpected weights), ``norm.gemma_add_rmsnorm_h6144.gemma_add_rmsnorm_mxfp8_for`` (codes in the
layout the consumer GEMM reads) and ``mxfp8_gemm_prequant`` (that GEMM): the same bytes as the explicit choice by
``prequant_code_tile_rows`` (decode row on (16, 16) codes, else row-major codes)."""

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8

from .test_gemm import require_gfx950

H = 6144


def _weight(n, k, seed):
    torch.manual_seed(seed)
    return quantize_mxfp8(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05)


@torch.inference_mode()
def test_gemm_warm_up_covers_the_norm_producer(monkeypatch):
    """After the K 6144 GEMM's load-time warm-up, the norm producer that writes its codes compiles nothing at any
    row count (first in the file: a fresh process has compiled nothing yet)."""
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as AN

    require_gfx950()
    w, ws = G.preshuffle_mxfp8_weight(*_weight(2560, H, 3))
    G.warmup_mxfp8_gemm_weight(w, ws, 64, torch.bfloat16)
    compiled = []
    real = MlirCompiler.compile
    monkeypatch.setattr(MlirCompiler, "compile", lambda self, *a, **kw: compiled.append(1) or real(self, *a, **kw))
    nw = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    for m in (1, 16, 33, 64, 1000, 4096):
        x = torch.randn((m, H), device="cuda", dtype=torch.bfloat16)
        AN.gemma_add_rmsnorm_mxfp8_for(x, x.clone(), nw, 1e-6, w)
    torch.cuda.synchronize()
    assert not compiled, len(compiled)


@torch.inference_mode()
def test_preshuffle_in_place():
    require_gfx950()
    w, ws = _weight(2560, H, 1)
    want_w, want_s = G.preshuffle_mxfp8_weight(w, ws)
    ptr_w, ptr_s = w.data_ptr(), ws.data_ptr()
    assert G.preshuffle_mxfp8_weight_(w, ws)
    assert (w.data_ptr(), ws.data_ptr()) == (ptr_w, ptr_s)
    assert torch.equal(w.view(torch.uint8), want_w.view(torch.uint8)) and torch.equal(ws, want_s)
    bad_w, bad_s = _weight(2560, H, 2)
    before = bad_w.clone()
    assert not G.preshuffle_mxfp8_weight_(bad_w, bad_s[:, :-1].contiguous())  # scale shape off
    assert not G.preshuffle_mxfp8_weight_(bad_w.view(torch.uint8), bad_s)  # not E4M3
    assert torch.equal(bad_w.view(torch.uint8), before.view(torch.uint8))


@pytest.mark.parametrize("n", [2560, 6144])
@pytest.mark.parametrize("m", [1, 7, 16, 64, 129, 256, 1000, 4096])
@torch.inference_mode()
def test_norm_for_gemm_then_prequant(m, n):
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as AN

    require_gfx950()
    w, ws = G.preshuffle_mxfp8_weight(*_weight(n, H, n + m))
    torch.manual_seed(m)
    x = torch.randn((m, H), device="cuda", dtype=torch.bfloat16)
    res = torch.randn((m, H), device="cuda", dtype=torch.bfloat16)
    nw = torch.randn(H, device="cuda", dtype=torch.bfloat16) * 0.1
    q, s, r = AN.gemma_add_rmsnorm_mxfp8_for(x, res, nw, 1e-6, w)
    got = G.mxfp8_gemm_prequant(q, s, w, ws, torch.bfloat16, m)
    tile = G.prequant_code_tile_rows(m, n, H, torch.bfloat16)
    if tile:
        q2, s2, r2 = AN.gemma_add_rmsnorm_mxfp8(x, res, nw, 1e-6, tile)
        want = G.mxfp8_gemm_shuffled(q2, s2, w, ws, torch.bfloat16, m)
    else:
        q2, s2, r2 = AN.gemma_add_rmsnorm_mxfp8_row_major(x, res, nw, 1e-6)
        want = G.mxfp8_gemm(q2, s2, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    assert torch.equal(r, r2)
    assert torch.equal(got.view(m, n), want.view(m, n))
    assert G.mxfp8_gemm_prequant(q[:0], s, w, ws, torch.bfloat16, 0).shape == (0, n)
