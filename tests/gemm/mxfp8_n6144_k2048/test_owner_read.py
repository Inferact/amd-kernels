# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Owned AGPR reads stay in the owning K-wave branch on measured o_proj rows."""
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize('m', [193, 194, 207, 224, 241, 255, 256, 321, 352, 384, 385, 416, 448, 449, 480, 512])
@pytest.mark.parametrize('spread', [False, True])
@torch.inference_mode()
def test_owned_accumulator_reads(m, spread):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 6144, 2048, seed=m)
    if spread:
        e = torch.randint(-5, 6, (m, 64), device='cuda')
        x = (x.float().view(m, 64, 32) * torch.exp2(e.float())[..., None]).view_as(x).bfloat16()
    x[0] = 0
    tile = g.prequant_code_tile_rows(m, 6144, 2048)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, tile)
    run = lambda: g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)
    got = run()
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    mt = ref.compare(got, expected)
    assert mt['cos'] >= ref.COS_MIN and mt['rel_l2'] <= ref.REL_L2_MAX, mt
    yf, rf = got.double(), expected.double()
    row_l2 = (yf - rf).norm(dim=1) / rf.norm(dim=1).clamp_min(1e-30)
    assert row_l2[1:].max() <= ref.REL_L2_MAX
    assert torch.nn.functional.cosine_similarity(yf[1:], rf[1:], dim=1).min() >= ref.COS_MIN
    assert not got[0].count_nonzero()
    assert torch.equal(got, g.mxfp8_linear(x, w, s))
    for _ in range(5):
        assert torch.equal(got, run())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for _ in range(3):
        captured.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got, captured)


def test_owned_read_scope():
    require_gfx950()
    for m in (192, 193, 256, 257, 320, 384, 512):
        c = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
        name = g._decode_launcher(6144, 2048, c, False).kernel_name
        assert name.endswith('_ownread') == (193 <= m <= 256 or 321 <= m <= 512), (m, name)


@torch.inference_mode()
def test_owned_read_next_to_mfma():
    from tests.mfma_spinner import mfma_spinner
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(256, 6144, 2048)
    a, s = g.mxfp8_e4m3_quantize_shuffled(x, 64)
    run = lambda: g.mxfp8_gemm_shuffled(a, s, w, ws, torch.bfloat16, 256)
    expected = run()
    spin = mfma_spinner(32)
    other = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, other)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, y) for y in results)


@pytest.mark.parametrize('m', [256, 512])
@torch.inference_mode()
def test_previous_owned_copy_override(m, monkeypatch):
    from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode
    require_gfx950()
    x, _, _, (w, s) = make_inputs(m, 6144, 2048)
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, g.prequant_code_tile_rows(m, 6144, 2048))
    expected = g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)

    def previous(n, k, c, a_bf16):
        return compile_mxfp8_gemm_decode(
            N=n, K=k, RT=c.tile_m // 16, NI=c.tile_n // 16, KW=c.waves_m,
            KS=c.split_k, LA=(c.la_w, c.la_a) if c.la_w else None,
            sched_barriers=bool(c.waves_n), W_CM=c.w_cm, A_BF16=a_bf16,
            OWNER_READ=False,
        )
    monkeypatch.setattr(g, '_decode_launcher', previous)
    assert torch.equal(expected, g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m))
