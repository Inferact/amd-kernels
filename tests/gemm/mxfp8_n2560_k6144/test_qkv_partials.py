# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 QKV's deferred split is tuned for the fused insert consumer.

The activation producer still uses prequant_code_tile_rows; standalone GEMMs and
other shapes retain their rows. Partials sum in a fixed order, and the capture
compiles every deferred row before replay.
"""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common import reference as ref

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize("m,sk,tm", [(1, 3, 16), (8, 3, 16), (9, 3, 16), (16, 3, 16), (17, 3, 32),
                                     (33, 3, 48), (48, 3, 48), (49, 3, 64), (64, 3, 64), (65, 3, 80), (80, 3, 80), (81, 3, 48), (96, 3, 48), (97, 3, 64), (128, 3, 64), (129, 2, 48), (144, 2, 48), (145, 2, 64), (176, 2, 64), (177, 2, 64), (192, 2, 64),
                                     (193, 2, 64), (256, 2, 64), (257, 2, 96), (288, 2, 96), (289, 1, 64)])
def test_deferred_qkv_dispatch(m, sk, tm):
    prequant = g._select_prequant_config(m, 2560, 6144, torch.bfloat16)
    deferred = g._deferred_qkv_config(m, 2560, 6144, prequant)
    assert deferred.split_k == sk
    assert max(16, deferred.tile_m) == tm
    # the deferred row reads a prefix of the producer's codes (its tile may be smaller)
    tile = g.prequant_code_tile_rows(m, 2560, 6144)
    assert -(-m // tm) * tm <= -(-m // tile) * tile
    for n, k in ((6144, 2048), (6144, 8192), (1536, 6144)):
        cfg = g._select_config(m, n, k, torch.bfloat16)
        assert g._deferred_qkv_config(m, n, k, cfg) is cfg


@pytest.mark.parametrize("m", [1, 4, 8, 13, 16, 17, 32, 33, 64, 65, 80, 96, 97, 111, 128, 129, 150, 176, 193, 225,
                               256, 257, 287, 288])
@torch.inference_mode()
def test_qkv_partials(m):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 2560, 6144, seed=m)
    if m > 1:
        x[0] = 0  # a zero row (a 1-row call keeps its row for the cosine)
    tile = g.prequant_code_tile_rows(m, 2560, 6144)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, tile)

    def launch():
        p = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m, defer_reduce=True)
        assert isinstance(p, g.SkPartials)
        return p

    part = launch()
    out = part.materialize()
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws,
                                      accumulate=torch.float64)
    metrics = ref.compare(out, expected)
    assert metrics["cos"] >= ref.COS_MIN
    assert metrics["rel_l2"] <= ref.REL_L2_MAX
    if m > 1:
        assert torch.count_nonzero(out[0]) == 0
    normal = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    variants = ref.compare(out, normal)
    assert variants["cos"] >= ref.COS_MIN and variants["rel_l2"] <= ref.REL_L2_MAX
    again = launch()
    assert torch.equal(part.tmp, again.tmp)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    captured.tmp.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured.tmp, part.tmp)


@torch.inference_mode()
def test_deferred_qkv_warmup(monkeypatch):
    import flydsl.compiler as flyc
    
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(289, 2560, 6144)
    g._decode_launcher.cache_clear()
    g._warmup_weight(w, ws, 289, torch.bfloat16)
    prepared = []
    for m in (1, 4, 8, 9, 12, 16, 17, 32, 33, 64, 65, 80, 96, 97, 111, 128, 129, 176, 177, 192, 193, 225, 256, 257,
              287, 288, 289):
        a, asc = g.mxfp8_e4m3_quantize_shuffled(x[:m], g.prequant_code_tile_rows(m, 2560, 6144))
        prepared.append((m, a, asc))
    compiled = []
    real = flyc.compile

    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for m, a, asc in prepared:
        g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m, defer_reduce=True)
        assert not compiled, f"M {m} compiled after warmup: {compiled}"
    torch.cuda.synchronize()
    assert not compiled


@pytest.mark.parametrize("m", [8, 9, 12, 15, 16, 17])
@torch.inference_mode()
def test_tiny_qkv_codes(m):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 2560, 6144, seed=m)
    x[0] = 0
    tile = g.prequant_code_tile_rows(m, 2560, 6144)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, tile)
    normal = g._select_config(m, 2560, 6144, torch.bfloat16)
    prequant = g._select_prequant_config(m, 2560, 6144, torch.bfloat16)
    if 9 <= m <= 16:
        assert prequant.waves_m == 1 and normal.waves_m == 4
        assert prequant.tile_m == normal.tile_m == 16
    else:
        assert prequant == normal
    got = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws,
                                      accumulate=torch.float64)
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX
    assert torch.count_nonzero(got[0]) == 0
    assert torch.equal(got, g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, captured)
