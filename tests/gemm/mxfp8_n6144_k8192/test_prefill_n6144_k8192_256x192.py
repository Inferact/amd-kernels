# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Full o_proj 256x192 prefill tile: ragged edges, FP64 reference, determinism and graph replay."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_n6144_k8192 import host as H_n6144_k8192
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

MS = [1281, 1311, 1408, 1535, 1536, 1537, 1664, 1792, 1920, 2047, 2048]


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_prefill_256x192_n6144_k8192(m, spread):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 6144, 8192, seed=m)
    if spread:
        exponents = torch.randint(-5, 6, (m, 8192 // 32), device="cuda")
        x = (x.float().view(m, -1, 32) * torch.exp2(exponents.float())[..., None]).view_as(x).bfloat16()
    x[m // 2] = 0
    assert g.prequant_code_tile_rows(m, 6144, 8192) == 0
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    call = lambda: g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    got = call()
    old_config = g._Config(256, 256, 128, 1, 1, 1, g._FOURWAVE_NT_KERNEL, 2, 2)
    old = g.mxfp8_gemm(a, asc, w, s, torch.bfloat16,
                      activation_scale_preshuffled=True, config=old_config)
    assert torch.equal(got, old)

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
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got, captured)


MS_K2048 = [3329, 3343, 3584, 3840, 3967, 3968]


@pytest.mark.parametrize("m", MS_K2048)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_prefill_256x192_n6144_k2048(m, spread):
    """The TP4 o_proj (K 2048) at 3329-3968 rows: the same tile compiled for K 2048."""
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 6144, 2048, seed=m)
    if spread:
        exponents = torch.randint(-5, 6, (m, 2048 // 32), device="cuda")
        x = (x.float().view(m, -1, 32) * torch.exp2(exponents.float())[..., None]).view_as(x).bfloat16()
    x[m // 2] = 0
    assert g._select_config(m, 6144, 2048).kernel == H_n6144_k8192._PREFILL_N6144_K8192_192_KERNEL
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    call = lambda: g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    got = call()
    four = g._Config(256, 256, 128, 1, 1, 1, g._FOURWAVE_NT_KERNEL, 2, 2)
    assert torch.equal(got, g.mxfp8_gemm(a, asc, w, s, torch.bfloat16,
                                         activation_scale_preshuffled=True, config=four))
    previous = g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True,
                            config=g._select_config(4096, 6144, 2048))
    assert ref.compare(got, previous)["cos"] >= ref.COS_MIN
    expected = ref.mxfp8_gemm_reference(*ref.quantize_mxfp8(x), wc, ws, accumulate=torch.float64)
    metrics = ref.compare(got, expected)
    assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, metrics
    row_g, row_r = got.float(), expected.float()
    valid = row_r.norm(dim=1) > 0
    assert torch.nn.functional.cosine_similarity(row_g[valid], row_r[valid]).min() >= ref.COS_MIN
    assert ((row_g - row_r).norm(dim=1) / row_r.norm(dim=1).clamp_min(1e-30)).max() <= ref.REL_L2_MAX
    assert torch.count_nonzero(got[m // 2]) == 0
    for _ in range(3):
        assert torch.equal(got, call())
    assert torch.equal(got, g.mxfp8_linear(x, w, s))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
    for _ in range(3):
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(got, captured)


@pytest.mark.parametrize("m", [3329, 3968])
@torch.inference_mode()
def test_prefill_256x192_k2048_dbo_plain_grid(m, monkeypatch):
    """Inside a DBO microbatch the K 2048 row launches one workgroup per tile (as the
    4-wave rows), with the persistent launch's bits."""
    require_gfx950()
    x, _, _, (w, s) = make_inputs(m, 6144, 2048, seed=m)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    call = lambda: g.mxfp8_gemm(a, asc, w, s, torch.bfloat16, activation_scale_preshuffled=True)
    persistent = call()
    monkeypatch.setattr(g, "DBO", True)
    plain = call()
    assert torch.equal(persistent, plain)
    assert H_n6144_k8192._prefill_n6144_k8192_192_launcher(2048, g._PLAIN_GRID_MAX).kernel_name.endswith(
        f"_k2048_nt_g{g._PLAIN_GRID_MAX}")


@torch.inference_mode()
def test_prefill_256x192_k2048_warmup(monkeypatch):
    import flydsl.compiler as flyc
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(3968, 6144, 2048)
    H_n6144_k8192._prefill_n6144_k8192_192_launcher.cache_clear()
    g._warmup_weight(w, ws, 4096, torch.bfloat16)
    assert H_n6144_k8192._prefill_n6144_k8192_192_launcher(2048).kernel_name == "flydsl_mxfp8_256x192_n6144_k2048_nt"
    prepared = [g.mxfp8_e4m3_quantize_aiter(x[:m]) for m in MS_K2048]
    compiled = []
    real = flyc.compile
    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(flyc, "compile", counting)
    for m, (a, asc) in zip(MS_K2048, prepared):
        g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
        g.mxfp8_linear(x[:m], w, ws)
        assert not compiled, (m, compiled)
    torch.cuda.synchronize()


@torch.inference_mode()
def test_prefill_256x192_warmup(monkeypatch):
    import flydsl.compiler as flyc
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(2048, 6144, 8192)
    H_n6144_k8192._prefill_n6144_k8192_192_launcher.cache_clear()
    H_n6144_k8192._prefill_n6144_k8192_192_one_launcher.cache_clear()
    g._warmup_weight(w, ws, 2048, torch.bfloat16)
    prepared = [g.mxfp8_e4m3_quantize_aiter(x[:m]) for m in MS]
    compiled = []
    real = flyc.compile
    def counting(*args, **kwargs):
        compiled.append(args[0])
        return real(*args, **kwargs)
    monkeypatch.setattr(flyc, "compile", counting)
    for m, (a, asc) in zip(MS, prepared):
        g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
        g.mxfp8_linear(x[:m], w, ws)
        assert not compiled, (m, compiled)
    torch.cuda.synchronize()


@pytest.mark.parametrize("m", [1281, 2048])
@torch.inference_mode()
def test_prefill_256x192_next_to_mfma(m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 6144, 8192)
    a, s = g.mxfp8_e4m3_quantize_aiter(x)
    run = lambda: g.mxfp8_gemm(a, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    expected = run()
    spin = mfma_spinner(32)
    other = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, other)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, got) for got in results)
