# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""QKV 192x256 prefill tile: ragged edges, FP64 reference, determinism and graph replay."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_n2560_k6144 import host as H_n2560_k6144
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

MS = [3073, 3263, 3264, 3265, 3457, 3584, 3839, 3840, 3841, 4095, 4096, 4097, 4417, 4608, 4799, 4800]


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_prefill_192x256_n2560_k6144(m, spread):
    require_gfx950()
    x, wc, ws, (w, s) = make_inputs(m, 2560, 6144, seed=m)
    if spread:
        exponents = torch.randint(-5, 6, (m, 6144 // 32), device="cuda")
        x = (x.float().view(m, -1, 32) * torch.exp2(exponents.float())[..., None]).view_as(x).bfloat16()
    x[m // 2] = 0
    assert g.prequant_code_tile_rows(m, 2560, 6144) == 0
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


@torch.inference_mode()
def test_prefill_192x256_warmup(monkeypatch):
    import flydsl.compiler as flyc
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(4800, 2560, 6144)
    H_n2560_k6144._prefill_n2560_k6144_192_launcher.cache_clear()
    g._warmup_weight(w, ws, 4800, torch.bfloat16)
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


@pytest.mark.parametrize("m", [3073, 4096])
@torch.inference_mode()
def test_prefill_192x256_next_to_mfma(m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 2560, 6144)
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
