# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""QKV rectangular tile: ragged edges, FP64 reference, determinism and graph replay."""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_n2560_k6144 import host as H_n2560_k6144
from ..mxfp8_common.test_gemm import make_inputs, require_gfx950

MS = [1537, 1599, 1600, 1601, 1664, 1791, 1792, 1793, 1921, 2047, 2048]


@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("spread", [False, True])
@torch.inference_mode()
def test_rect_n2560_k6144(m, spread):
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
def test_rect_warmup(monkeypatch):
    import flydsl.compiler as flyc
    require_gfx950()
    x, _, _, (w, ws) = make_inputs(2048, 2560, 6144)
    H_n2560_k6144._rect_n2560_k6144_launcher.cache_clear()
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


@pytest.mark.parametrize('m', [1537, 1791, 2048])
@torch.inference_mode()
def test_rect_qkv_terminal_output_guards(m):
    from amd_kernels.common.tensor_shim import _run_compiled

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 2560, 6144, seed=m)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    expected = g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    arena = torch.empty((m + 2, 2560), dtype=torch.bfloat16, device='cuda')
    arena.view(torch.int16).fill_(0x3555)
    guard0, guard1 = arena[0].clone(), arena[-1].clone()
    out = arena[1:-1]
    launch = H_n2560_k6144._rect_n2560_k6144_launcher()
    def run():
        _run_compiled(launch, a.view(torch.int8).view(-1), w.view(torch.int8).view(-1), out.view(-1),
                      asc.view(torch.uint8).view(-1), ws.view(torch.uint8).view(-1),
                      m, 2560, torch.cuda.current_stream())
    run()
    assert torch.equal(out, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(8):
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected)
        assert torch.equal(arena[0], guard0) and torch.equal(arena[-1], guard1)


@pytest.mark.parametrize('m', [1537, 2048])
@torch.inference_mode()
def test_rect_qkv_terminal_next_to_mfma(m):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    x, _, _, (w, ws) = make_inputs(m, 2560, 6144, seed=m)
    a, asc = g.mxfp8_e4m3_quantize_aiter(x)
    run = lambda: g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    expected = run()
    spin = mfma_spinner(32)
    side = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    spin(1_000_000, side)
    results = [run() for _ in range(16)]
    torch.cuda.synchronize()
    assert all(torch.equal(expected, out) for out in results)
