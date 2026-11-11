# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The TP4 o_proj's 128x192 tile (N 6144 x K 2048, 769-1024 rows): ragged edges, FP64 reference, determinism and graph
replay."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_n6144_k2048 import host as H_n6144_k2048
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

K = 2048
# The TP4 o_proj window: 769-1024 rows.
MS = [769, 770, 800, 831, 896, 897, 960, 1023, 1024]


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_tile_128x192(m, spread):
    require_gfx950()
    k = K
    x, wc, ws, (w, s) = make_inputs(m, 6144, k, seed=m)
    if spread:
        exponents = torch.randint(-5, 6, (m, k // 32), device="cuda")
        x = (x.float().view(m, -1, 32) * torch.exp2(exponents.float())[..., None]).view_as(x).bfloat16()
    x[m // 2] = 0
    assert g._select_config(m, 6144, k, torch.bfloat16).kernel == "n6144_k2048_128x192"
    assert g.prequant_code_tile_rows(m, 6144, k) == 0
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    call = lambda: g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    got = call()
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, metrics
    row_g, row_r = got.float(), expected.float()
    valid = row_r.norm(dim=1) > 0
    assert torch.nn.functional.cosine_similarity(row_g[valid], row_r[valid]).min() >= ref.COS_MIN
    assert ((row_g - row_r).norm(dim=1) / row_r.norm(dim=1).clamp_min(1e-30)).max() <= ref.REL_L2_MAX
    assert torch.count_nonzero(got[m // 2]) == 0
    for _ in range(5):
        assert torch.equal(got, call())
    assert torch.equal(got, g.mxfp8_linear(x, w, s))
    assert torch.equal(got, g.mxfp8_gemm(*ref.quantize_mxfp8(x), w, s, torch.bfloat16))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got, captured)


@torch.inference_mode()
def test_tile_128x192_warmup(monkeypatch):
    import flydsl.compiler as flyc
    require_gfx950()
    x2, _, _, (w2, ws2) = make_inputs(1024, 6144, K)
    H_n6144_k2048._n6144_k2048_128_launcher.cache_clear()
    g._warmup_weight(w2, ws2, 1024, torch.bfloat16)
    prepared = [(x2, w2, ws2, m, g.mxfp8_e4m3_quantize_aiter(x2[:m])) for m in MS]
    compiled = []
    real = flyc.compile
    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(flyc, "compile", counting)
    for xx, ww, wws, m, (a, asc) in prepared:
        g.mxfp8_gemm(a, asc, ww, wws, torch.bfloat16, activation_scale_preshuffled=True)
        g.mxfp8_linear(xx[:m], ww, wws)
        assert not compiled, (m, compiled)
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [769, 1024])
@torch.inference_mode()
def test_n6144_k2048_terminal_ring_next_to_mfma(m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 6144, 2048, seed=m)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    run = lambda: g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    expected = run()
    spin = mfma_spinner(32)
    side = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, side)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, got) for got in results)
