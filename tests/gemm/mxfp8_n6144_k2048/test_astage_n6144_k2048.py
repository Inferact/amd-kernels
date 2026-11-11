# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The immutable-A/scale 64x48 kernel at o_proj's 97-128-row window."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref

from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize("m", [97, 98, 111, 112, 113, 120, 127, 128])
@torch.inference_mode()
def test_astage_reference_and_poisoned_replay(m):
    require_gfx950()
    x, wc, ws, (w, scales) = make_inputs(m, 6144, 2048, seed=m)
    torch.manual_seed(1000 + m)
    shifts = torch.randint(-5, 6, (m, 64, 1), device=x.device)
    x = (x.float().view(m, 64, 32) * torch.exp2(shifts.float())).view(m, 2048).bfloat16()
    x[0].zero_()
    assert g.prequant_code_tile_rows(m, 6144, 2048) == 64
    c, s = g.mxfp8_e4m3_quantize_shuffled(x, 64)

    def run():
        return g.mxfp8_gemm_shuffled(c, s, w, scales, torch.bfloat16, m)

    got = run()
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    mt = ref.compare(got, expected)
    assert mt["cos"] >= ref.COS_MIN and mt["rel_l2"] <= ref.REL_L2_MAX, mt
    active = expected.double().norm(dim=1) > 0
    row_l2 = (got.double() - expected.double()).norm(dim=1) / expected.double().norm(dim=1).clamp_min(1e-30)
    row_cos = torch.nn.functional.cosine_similarity(got.double()[active], expected.double()[active], dim=1)
    assert row_l2[active].max() <= ref.REL_L2_MAX
    assert row_cos.min() >= ref.COS_MIN
    assert got[0].count_nonzero() == 0
    assert torch.equal(got, run())
    assert torch.equal(got, g.mxfp8_linear(x, w, scales, torch.bfloat16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = run()
    for _ in range(3):
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, got)


def test_astage_variant_scope():
    for m in (96, 97, 128, 129):
        cfg = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
        name = g._decode_launcher(6144, 2048, cfg, False).kernel_name
        assert ("astage_n6144_k2048" in name) == (97 <= m <= 128), (m, name)
    cfg = g._select_prequant_config(128, 6144, 8192, torch.bfloat16)
    assert "astage_n6144_k2048" not in g._decode_launcher(6144, 8192, cfg, False).kernel_name


@torch.inference_mode()
def test_astage_public_warmup(monkeypatch):
    import flydsl.compiler as flyc

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(192, 6144, 2048)
    g._decode_launcher.cache_clear()
    g._warmup_weight(w, ws, 192, torch.bfloat16)
    cases = []
    for m in (96, 97, 99, 111, 112, 113, 127, 128, 129):
        c, s = g.mxfp8_e4m3_quantize_shuffled(x[:m], g.prequant_code_tile_rows(m, 6144, 2048))
        cases.append((m, c, s))
    compiled = []
    real = flyc.compile

    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for m, c, s in cases:
        g.mxfp8_gemm_shuffled(c, s, w, ws, torch.bfloat16, m)
        g.mxfp8_linear(x[:m], w, ws, torch.bfloat16)
        assert not compiled, (m, compiled)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_astage_next_to_mfma():
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(128, 6144, 2048)
    c, s = g.mxfp8_e4m3_quantize_shuffled(x, 64)
    expected = g.mxfp8_gemm_shuffled(c, s, w, ws, torch.bfloat16, 128)
    spin = mfma_spinner(32)
    other = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, other)
    results = [g.mxfp8_gemm_shuffled(c, s, w, ws, torch.bfloat16, 128) for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, got) for got in results)
