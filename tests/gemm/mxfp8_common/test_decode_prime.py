# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Interleaved prime on deferred 64x64 split-3: same slabs and graph-replay bytes."""
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common import mxfp8_gemm_decode as d
from .test_gemm import make_inputs, require_gfx950


@pytest.mark.parametrize('m', (97, 98, 104, 112, 113, 127, 128))
@torch.inference_mode()
def test_interleaved_prime_bitwise(m, monkeypatch):
    require_gfx950()
    x, _, _, (w, s) = make_inputs(m, 2560, 6144, seed=m + 17)
    x[2].zero_()
    gen = torch.Generator(device='cuda').manual_seed(m)
    spread = torch.randint(-5, 6, (m, 192), device='cuda', generator=gen)
    x = (x.float().view(m, 192, 32) * torch.exp2(spread.float())[:, :, None]).view(m, 6144).bfloat16()
    c = g._deferred_qkv_config(m, 2560, 6144, g._select_prequant_config(m, 2560, 6144, torch.bfloat16))
    old = d.compile_mxfp8_gemm_decode(
        N=2560, K=6144, RT=c.tile_m // 16, NI=c.tile_n // 16, KW=c.waves_m, KS=c.split_k,
        sched_barriers=bool(c.waves_n), W_CM=c.w_cm, INTERLEAVE_PRIME=False,
    )
    new = g._decode_launcher(2560, 6144, c, False)
    assert '_ip_ont' in new.kernel_name and '_ip' not in old.kernel_name
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, g.prequant_code_tile_rows(m, 2560, 6144))

    def run():
        return g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m, defer_reduce=True).tmp

    got = run().clone()
    assert torch.equal(got, run())
    with monkeypatch.context() as mp:
        mp.setattr(g, '_decode_launcher', lambda *args: old)
        assert torch.equal(got, run())
    assert got[:, 2].count_nonzero() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for _ in range(3):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)


@pytest.mark.parametrize('m', (8, 32, 80, 81, 96, 129, 144, 192, 256, 288, 512))
def test_prime_does_not_change_other_rows(m):
    c = g._deferred_qkv_config(m, 2560, 6144, g._select_prequant_config(m, 2560, 6144, torch.bfloat16))
    assert '_ip' not in g._decode_launcher(2560, 6144, c, False).kernel_name


@pytest.mark.parametrize('m', (257, 271, 288, 319, 320, 385, 399, 416, 447, 448))
@torch.inference_mode()
def test_o_proj_a_first_prime_bitwise(m, monkeypatch):
    require_gfx950()
    x, _, _, (w, s) = make_inputs(m, 6144, 2048, seed=m + 17)
    x[2].zero_()
    gen = torch.Generator(device='cuda').manual_seed(m)
    spread = torch.randint(-5, 6, (m, 64), device='cuda', generator=gen)
    x = (x.float().view(m, 64, 32) * torch.exp2(spread.float())[:, :, None]).view(m, 2048).bfloat16()
    c = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
    old = d.compile_mxfp8_gemm_decode(
        N=6144, K=2048, RT=c.tile_m // 16, NI=c.tile_n // 16, KW=c.waves_m, KS=c.split_k,
        sched_barriers=bool(c.waves_n), W_CM=c.w_cm, INTERLEAVE_PRIME=False,
    )
    assert '_ip' in g._decode_launcher(6144, 2048, c, False).kernel_name
    a, asc = g.mxfp8_e4m3_quantize_shuffled(x, g.prequant_code_tile_rows(m, 6144, 2048))

    def run():
        return g.mxfp8_gemm_shuffled(a, asc, w, s, torch.bfloat16, m)

    got = run().clone()
    assert torch.equal(got, run())
    with monkeypatch.context() as mp:
        mp.setattr(g, '_decode_launcher', lambda *args: old)
        assert torch.equal(got, run())
    assert got[2].count_nonzero() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for _ in range(3):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, got)


@pytest.mark.parametrize('m', (32, 48, 64, 96, 128, 160, 192, 193, 256, 321, 384, 449, 512))
def test_o_proj_other_prime_windows(m):
    c = g._select_prequant_config(m, 6144, 2048, torch.bfloat16)
    assert '_ip' not in g._decode_launcher(6144, 2048, c, False).kernel_name


@pytest.mark.parametrize('m,n,k,defer', ((128, 2560, 6144, True), (320, 6144, 2048, False),
                                      (448, 6144, 2048, False), (512, 6144, 2048, False)))
@torch.inference_mode()
def test_prime_next_to_mfma(m, n, k, defer):
    from tests.mfma_spinner import mfma_spinner
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, n, k)
    a, s = g.mxfp8_e4m3_quantize_shuffled(x, g.prequant_code_tile_rows(m, n, k))

    def run():
        y = g.mxfp8_gemm_shuffled(a, s, w, ws, torch.bfloat16, m, defer_reduce=defer)
        return y.tmp if defer else y

    expected = run().clone()
    spin = mfma_spinner(32)
    other = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, other)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, y) for y in results)
