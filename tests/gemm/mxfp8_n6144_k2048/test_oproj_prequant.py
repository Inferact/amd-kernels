# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Tiny o_proj: codes avoid K-wave reduction; BF16 retains in-kernel quantization."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950, unshuffle_scales
from aiter.ops.shuffle import shuffle_weight


@pytest.mark.parametrize('m', [1, 4, 8, 9, 16, 17, 24, 31, 32, 33])
@torch.inference_mode()
def test_tiny_oproj_prequant(m):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 6144, 2048, seed=m)
    x[0] = 0
    normal = g._select_config(m, 6144, 2048, torch.bfloat16)
    new = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
    tile = g.prequant_code_tile_rows(m, 6144, 2048)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, tile)
    old_a, old_asc = g.mxfp8_e4m3_quantize_shuffled(x, max(16, normal.tile_m))
    # The 17-32 window changes the tile to 32, but the producer's bytes/shape stay the same.
    mask = torch.zeros_like(a, dtype=torch.uint8)
    mask[:m] = 1
    mask = shuffle_weight(mask, layout=(16, 16)).contiguous().bool()
    assert torch.equal(a.view(torch.uint8)[mask], old_a.view(torch.uint8)[mask])
    assert torch.equal(unshuffle_scales(asc, m, 64), unshuffle_scales(old_asc, m, 64))
    out = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    old = g.mxfp8_gemm_shuffled(old_a, old_asc, w, s, torch.bfloat16, m, config=normal)
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws,
                                      accumulate=torch.float64)
    if m > 1:
        metrics = ref.compare(out, expected)
        assert metrics['cos'] >= ref.COS_MIN and metrics['rel_l2'] <= ref.REL_L2_MAX
    if m > 1:
        variants = ref.compare(out, old)
        assert variants['cos'] >= ref.COS_MIN and variants['rel_l2'] <= ref.REL_L2_MAX
    else:
        assert torch.equal(out, old)
    assert torch.count_nonzero(out[0]) == 0
    if m <= 32:
        assert new.kernel == 'decode' and new.waves_m == 1
        assert normal.kernel == 'decode_fq'
    else:
        assert new == normal
    # The BF16 entry keeps its selected fused-quant row.
    linear = g.mxfp8_linear(x, w, s)
    assert torch.equal(linear.view(torch.uint8), g.mxfp8_linear(x, w, s, config=normal).view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    replayed.fill_(float('nan'))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed.view(torch.uint8), out.view(torch.uint8))


@torch.inference_mode()
def test_tiny_oproj_warmup(monkeypatch):
    import flydsl.compiler as flyc
    
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(33, 6144, 2048)
    g._decode_launcher.cache_clear()
    g._warmup_weight(w, ws, 33, torch.bfloat16)
    inputs = [(m, g.mxfp8_e4m3_quantize_shuffled(x[:m], g.prequant_code_tile_rows(m, 6144, 2048)))
              for m in (1, 4, 8, 9, 16, 17, 24, 31, 32, 33)]
    calls = []
    real = flyc.compile

    def count(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(flyc, 'compile', count)
    for m, (a, asc) in inputs:
        g.mxfp8_gemm_shuffled(a, asc, w, ws, torch.bfloat16, m)
    torch.cuda.synchronize()
    assert not calls
