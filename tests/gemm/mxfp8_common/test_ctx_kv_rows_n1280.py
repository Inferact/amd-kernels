# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The bf16 rows of N 1280 x K 6144 (the TP4 context-K / V GEMM). Each row keeps the split-K of the generic AITER
fallback (split-K 24 / 8 / 6 / 4 by M below 257, 128x128 / 128x256 split-K 1 above) with one K-wave (decode) or its
AITER tile_k (1025-2048), so every output is bitwise the fallback's: checked here, with the fp64 contract of
reference.py, graph replay, the bf16 entry, no compile after the warm-up and MFMA co-residency. Rows 1-64 and above
2048 use the fallback."""
import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference, quantize_mxfp8

N, K = 1280, 6144
# (first M, last M, kernel, tile_m, tile_n, split_k)
ROWS = [
    (65, 80, "decode", 16, 64, 8),
    (81, 128, "decode", 32, 32, 8),
    (129, 192, "decode", 48, 64, 6),
    (193, 256, "decode", 48, 64, 4),
    (257, 512, "decode", 80, 32, 1),
    (513, 1024, "decode", 176, 32, 1),
    (1025, 2048, "aiter", 32, 128, 1),
]


def fallback(m):
    """gemm._select_config's generic rule for an untuned (N, K): the configuration before these rows."""
    if m <= 16:
        tm, tn, wpe, xcd = 32, 64, 0, 0
    elif m <= 256:
        tm, tn, wpe, xcd = 64, 64, 0, 0
    elif m <= 1024:
        tm, tn, wpe, xcd = 128, 128, 2, 4
    else:
        tm, tn, wpe, xcd = 128, 256, 2, 4
    sk = 1
    if m <= 256:
        need = -(-256 // (-(-m // tm) * (N // tn)))
        sk = next(c for c in range(need, min(32, K // 256) + 1) if K % (c * 256) == 0)
    return G._Config(tm, tn, 128, wpe, xcd, sk)


def test_dispatch():
    for lo, hi, kern, tm, tn, sk in ROWS:
        for m in sorted({lo, (lo + hi) // 2, hi}):
            c = G._select_config(m, N, K)
            assert (c.kernel, c.tile_m, c.tile_n, c.split_k) == (kern, tm, tn, sk), (m, c)
            assert c.split_k == fallback(m).split_k
            if kern == "decode":
                assert c.waves_m == 1 and G.prequant_code_tile_rows(m, N, K) == tm
            else:
                assert c.tile_k == fallback(m).tile_k and G.prequant_code_tile_rows(m, N, K) == 0
    for m in (1, 8, 16, 17, 32, 48, 49, 64, 2049, 4096):
        assert G._select_config(m, N, K) == fallback(m), m
    assert G._select_config(300, N, K, torch.float16) == fallback(300)


def _spread(shape, gen):
    rows, cols = shape
    x = torch.randn(shape, generator=gen, device="cuda")
    e = torch.randint(-5, 6, (rows, cols // 32, 1), generator=gen, device="cuda")
    return x * torch.exp2(e.expand(rows, cols // 32, 32).reshape(rows, cols).float())


@pytest.fixture(scope="module")
def weights():
    gen = torch.Generator(device="cuda").manual_seed(1280)
    q, s = quantize_mxfp8((_spread((N, K), gen) * 0.02).bfloat16())
    return (q, s, *G.preshuffle_mxfp8_weight(q, s))


def _entry(x, w, ws, m):
    """the call a fused producer makes: codes in the row's layout, then the shuffled or row-major entry"""
    tile = G.prequant_code_tile_rows(m, N, K)
    if tile:
        q, s = G.mxfp8_e4m3_quantize_shuffled(x, tile)
        return q, s, lambda: G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m)
    q, s = G.mxfp8_e4m3_quantize_aiter(x)
    return q, s, lambda: G.mxfp8_gemm(q, s, w, ws, torch.bfloat16, activation_scale_preshuffled=True)


@pytest.mark.parametrize("m", (65, 79, 80, 81, 100, 128, 129, 160, 193, 256, 257, 300, 511, 513,
                               700, 1024, 1025, 1500, 2048))
def test_fallback_bits_fp64_graph(weights, m):
    qw, sw, w, ws = weights
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = _spread((m, K), gen).bfloat16()
    x[0].zero_()
    qa, sa = G.mxfp8_e4m3_quantize_aiter(x)
    ref = torch.full((m, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    G._launch_tiled(fallback(m), qa, sa, w, ws, ref)
    q, s, call = _entry(x, w, ws, m)
    out = call()
    torch.cuda.synchronize()
    assert torch.equal(out, ref), (m, int((out != ref).sum()))
    assert torch.equal(G.mxfp8_linear(x, w, ws, torch.bfloat16), ref)
    qx, sx = quantize_mxfp8(x)
    exact = mxfp8_gemm_reference(qx, sx, qw, sw, accumulate=torch.float64).double()
    assert not out[0].any()
    cos = torch.nn.functional.cosine_similarity(out[1:].double(), exact[1:], dim=1)
    rel = (out[1:].double() - exact[1:]).norm(dim=1) / exact[1:].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3, (m, float(cos.min()), float(rel.max()))
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        gout = call()
    gout.fill_(float("nan"))
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(gout, ref)
    x2 = _spread((m, K), gen).bfloat16()
    q2, s2, call2 = _entry(x2, w, ws, m)
    q.copy_(q2)
    s.copy_(s2)
    gr.replay()
    eager = call2()
    torch.cuda.synchronize()
    assert torch.equal(gout, eager)


def test_no_compile_after_warmup(weights):
    from flydsl.compiler.jit_function import MlirCompiler

    _, _, w, ws = weights
    zero = lambda m: torch.zeros((m, K), device="cuda", dtype=torch.bfloat16)  # noqa: E731
    calls = []  # the codes are made before the trap: only the GEMM rows are under test (the producer is the norm)
    for lo, hi, *_ in ROWS:
        _entry(zero(hi), w, ws, hi)[2]()
        calls += [_entry(zero(m), w, ws, m)[2] for m in sorted({lo, lo + 1, (lo + hi) // 2, hi - 1})]
    torch.cuda.synchronize()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for call in calls:
            call()
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig


@pytest.mark.parametrize("m", [r[1] for r in ROWS])
def test_mfma_co_residency(weights, m):
    from tests.mfma_spinner import mfma_spinner

    _, _, w, ws = weights
    _, _, call = _entry(torch.randn((m, K), device="cuda", dtype=torch.bfloat16), w, ws, m)
    ref = call()
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for _ in range(3):
        spinner(100_000, side)
        got = call()
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
