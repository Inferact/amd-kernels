# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 QKV (N 2560, K 6144) at 385-1536 rows: decode rows with 80-column tiles.

N 2560 is 32 tiles of 80 columns, so eight row tiles fill the 256 CUs with one workgroup
each (row tile 64 / 80 / 96 / 112 / 128 / 144 / 160 / 176 / 192 by window; 513-576 keeps the
96x64 row). The producer writes the codes in the row's tile (``prequant_code_tile_rows``);
``mxfp8_linear`` quantizes into it itself. Against the fp64 product of the same quantized
operands at the window edges, with a zero row; bitwise repeats; graph replay over a
NaN-poisoned output; no compile after the warm-up.
"""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common import reference as ref

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

EDGES = [385, 448, 449, 512, 513, 576, 577, 640, 641, 672, 673, 768, 769, 770, 784, 785, 895, 896, 897, 1000, 1024, 1025, 1152, 1153, 1280,
         1281, 1408, 1409, 1536]


@pytest.mark.parametrize("m", EDGES)
@torch.inference_mode()
def test_qkv_bn80_rows(m):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 2560, 6144, seed=m)
    x[m // 2] = 0
    tile = g.prequant_code_tile_rows(m, 2560, 6144)
    assert tile == g._select_config(m, 2560, 6144, torch.bfloat16).tile_m
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, tile)
    got = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, metrics
    # The grouped XCD order must cover every row even when its last row group
    # has three row tiles (and an odd-RT tile repairs the scale-word pairing).
    active = expected.double().norm(dim=1) > 0
    row_l2 = (got.double() - expected.double()).norm(dim=1) / expected.double().norm(dim=1).clamp_min(1e-30)
    assert row_l2[active].max() <= ref.REL_L2_MAX
    assert torch.count_nonzero(got[m // 2]) == 0
    assert torch.equal(got, g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m))
    # the bf16 entry quantizes into the same layout itself: same bytes
    assert torch.equal(got, g.mxfp8_linear(x, w, s, torch.bfloat16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(got, captured)


@torch.inference_mode()
def test_qkv_bn80_warmup(monkeypatch):
    import flydsl.compiler as flyc
    
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(1536, 2560, 6144)
    g._decode_launcher.cache_clear()
    g._warmup_weight(w, ws, 1536, torch.bfloat16)
    prepared = []
    for m in (*EDGES, 400, 700, 900, 1111, 1234, 1350, 1500):
        a, asc = g.mxfp8_e4m3_quantize_shuffled(x[:m], g.prequant_code_tile_rows(m, 2560, 6144))
        prepared.append((m, a, asc))
    compiled = []
    real = flyc.compile

    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for m, a, asc in prepared:
        g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m)
        g.mxfp8_linear(x[:m], w, ws, torch.bfloat16)
        assert not compiled, f"M {m} compiled after warmup: {compiled}"
    torch.cuda.synchronize()
    assert not compiled
