# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""N 2560 x K 6144 serves two layers: the QKV projection (N 2560 = 16 q + 1 k + 1 v + 2 index heads) and the draft
context K / V projection (N 2560 = 5 layers x K, V x 2 kv heads x 128). The QKV-only hooks of its shape host
(``select_prequant`` at 9-16 rows, ``select_deferred`` for the norm / RoPE / insert chain) must hold for the context
K / V too: the producer's code tile is the one the GEMM reads, the codes entry and the deferred split-K partials meet
the fp64 contract per output row, and a deferred row reads a prefix of the producer's codes."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference, quantize_mxfp8

from .test_gemm import require_gfx950
from .test_shared_expert_rows import _spread

N, K = 2560, 6144
MS = [1, 7, 8, 9, 14, 16, 17, 56, 64, 80, 112, 128, 144, 192, 224, 256, 288, 416, 512, 1024, 2048, 3072, 4096]


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(N + K)
    q, s = quantize_mxfp8((_spread((N, K), gen) * 0.02).bfloat16())
    return q, s, *G.preshuffle_mxfp8_weight(q, s)


def _check(out, exact, tag):
    lo = 1 if out.shape[0] > 1 else 0  # row 0 is the zero row
    got = out[lo:].double()
    cos = torch.nn.functional.cosine_similarity(got, exact[lo:], dim=1)
    rel = (got - exact[lo:]).norm(dim=1) / exact[lo:].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3, (tag, float(cos.min()), float(rel.max()))


def test_code_tile_is_the_rows_tile():
    for m in range(1, 4097):
        tile = G.prequant_code_tile_rows(m, N, K)
        cfg = G._select_prequant_config(m, N, K, torch.bfloat16)
        if cfg.kernel in ("decode", "decode_fq"):
            assert tile == max(16, cfg.tile_m), m
            deferred = G._deferred_qkv_config(m, N, K, cfg)
            rows = -(-m // deferred.tile_m) * deferred.tile_m
            assert rows <= -(-m // tile) * tile, m  # a deferred row reads a prefix of the producer's codes
        else:
            assert tile == 0, m


@pytest.mark.parametrize("m", MS)
@torch.inference_mode()
def test_codes_and_deferred_entries(weights, m):
    require_gfx950()
    qw, sw, w, ws = weights
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = _spread((m, K), gen).bfloat16()
    if m > 1:
        x[0].zero_()
    exact = mxfp8_gemm_reference(*quantize_mxfp8(x), qw, sw, accumulate=torch.float64).double()
    linear = G.mxfp8_linear(x, w, ws, torch.bfloat16)
    _check(linear, exact, ("linear", m))
    tile = G.prequant_code_tile_rows(m, N, K)
    if not tile:
        a, s = G.mxfp8_e4m3_quantize_aiter(x)
        _check(G.mxfp8_gemm_prequant(a, s, w, ws, torch.bfloat16, m), exact, ("codes", m))
        return
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, tile)
    codes = G.mxfp8_gemm_prequant(q, s, w, ws, torch.bfloat16, m)
    _check(codes, exact, ("codes", m))
    deferred = G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m, defer_reduce=True)
    if isinstance(deferred, G.SkPartials):
        deferred = deferred.materialize()
    _check(deferred, exact, ("deferred", m))
    if m > 1:
        assert not codes[0].any() and not deferred[0].any()
