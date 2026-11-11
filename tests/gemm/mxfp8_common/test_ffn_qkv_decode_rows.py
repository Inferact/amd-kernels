# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP4 decode rows of N 2304 x K 6144 (QKV), N 6144 x K 6144 (gate_up) and N 6144 x K 3072 (down): QKV 97-112
(32x32, K-waves 2, cached W), 193-224 (80x32), 385-448 (cached W), gate_up 193-224 (112x48), down 65-80 (48x48) and
193-224 (112x48); each keeps the K-wave count of a baseline tile (``ROWS``' last column), so its K order is the
baseline's: bitwise equal to the baseline tile, the fp64 contract of reference.py, graph replay, the bf16 entry, no
compile after the warm-up, MFMA co-residency. Neighbouring rows are pinned."""
import dataclasses

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference, quantize_mxfp8

# (N, K, first M, last M, row (tile_m, tile_n, K-waves, w_cm), baseline (tile_m, tile_n, w_cm))
ROWS = [
    (2304, 6144, 97, 112, (32, 32, 2, 0), (32, 48, None)),
    (2304, 6144, 193, 224, (80, 32, 3, None), (64, 48, None)),
    (2304, 6144, 385, 448, (64, 64, 3, 0), (64, 64, None)),
    (6144, 6144, 193, 224, (112, 48, 3, None), (128, 48, None)),
    (6144, 3072, 65, 80, (48, 48, 3, None), (64, 48, None)),
    (6144, 3072, 193, 224, (112, 48, 3, None), (128, 48, None)),
]
# the rows just outside each range: (N, K, M, (tile_m, tile_n, K-waves, w_cm))
NEIGHBOURS = [
    (2304, 6144, 96, (32, 32, 1, None)), (2304, 6144, 113, (32, 48, 2, None)), (2304, 6144, 160, (32, 48, 2, None)),
    (2304, 6144, 192, (64, 32, 4, None)), (2304, 6144, 225, (64, 48, 3, None)), (2304, 6144, 320, (64, 48, 3, None)),
    (2304, 6144, 384, (128, 32, 3, None)), (2304, 6144, 449, (128, 48, 3, None)),
    (6144, 6144, 192, (96, 48, 4, None)), (6144, 6144, 225, (128, 48, 3, None)), (6144, 6144, 256, (128, 48, 3, None)),
    (6144, 3072, 64, (64, 32, 1, None)), (6144, 3072, 81, (64, 48, 3, None)), (6144, 3072, 128, (64, 48, 3, None)),
    (6144, 3072, 192, (96, 48, 3, None)), (6144, 3072, 225, (128, 48, 3, None)), (6144, 3072, 256, (128, 48, 3, None)),
]


def _key(cfg):
    return (cfg.tile_m, cfg.tile_n, cfg.waves_m, cfg.w_cm)


def test_dispatch_edges():
    for n, k, lo, hi, new, _ in ROWS:
        for m in sorted({lo, lo + 1, (lo + hi) // 2, hi - 1, hi}):
            cfg = G._select_config(m, n, k)
            assert cfg.kernel == "decode" and cfg.split_k == 1 and _key(cfg) == new, (n, k, m, cfg)
            assert G.prequant_code_tile_rows(m, n, k) == new[0]
            assert G._select_config(m, n, k, torch.float16).kernel == "aiter"
    for n, k, m, old in NEIGHBOURS:
        assert _key(G._select_config(m, n, k)) == old, (n, k, m)


def _spread(shape, gen):
    """normal values with per-32 scales spread over +-5 binades"""
    rows, cols = shape
    x = torch.randn(shape, generator=gen, device="cuda")
    e = torch.randint(-5, 6, (rows, cols // 32, 1), generator=gen, device="cuda")
    return x * torch.exp2(e.expand(rows, cols // 32, 32).reshape(rows, cols).float())


_WEIGHTS = {}


def _weights(n, k):
    if (n, k) not in _WEIGHTS:
        gen = torch.Generator(device="cuda").manual_seed(n + k)
        q, s = quantize_mxfp8((_spread((n, k), gen) * 0.02).bfloat16())
        _WEIGHTS[(n, k)] = (q, s, *G.preshuffle_mxfp8_weight(q, s))
    return _WEIGHTS[(n, k)]


CASES = []
for _n, _k, _lo, _hi, _new, _old in ROWS:
    for _m in sorted({_lo, _lo + 5, (_lo + _hi) // 2 + 3, _hi}):
        CASES.append((_n, _k, _m, _old))


@pytest.mark.parametrize("n,k,m,old", CASES)
def test_old_order_fp64_graph(n, k, m, old):
    qw, sw, w, ws = _weights(n, k)
    gen = torch.Generator(device="cuda").manual_seed(m)
    x = _spread((m, k), gen).bfloat16()
    x[0].zero_()
    cfg = G._select_config(m, n, k)
    old_cfg = dataclasses.replace(cfg, tile_m=old[0], tile_n=old[1], w_cm=old[2])
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    ref = torch.full_like(out, float("nan"))
    G._launch_decode(old_cfg, *G.mxfp8_e4m3_quantize_shuffled(x, old[0]), w, ws, ref)
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, cfg.tile_m)
    G._launch_decode(cfg, q, s, w, ws, out)
    torch.cuda.synchronize()
    assert torch.equal(out, ref), (n, k, m, int((out != ref).sum()))
    qx, sx = quantize_mxfp8(x)
    exact = mxfp8_gemm_reference(qx, sx, qw, sw, accumulate=torch.float64).double()
    assert not out[0].any()
    got = out[1:].double()
    cos = torch.nn.functional.cosine_similarity(got, exact[1:], dim=1)
    rel = (got - exact[1:]).norm(dim=1) / exact[1:].norm(dim=1)
    assert cos.min() >= 0.99999 and rel.max() <= 1e-3, (n, k, m, float(cos.min()), float(rel.max()))
    total = (out.double() - exact).norm() / exact.norm()
    assert total <= 1e-3
    # the entry a fused producer calls, under graph replay with a poisoned output and a changed input
    shuffled = lambda: G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m)  # noqa: E731
    assert torch.equal(shuffled(), ref)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        gout = shuffled()
    gout.fill_(float("nan"))
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(gout, ref)
    x2 = _spread((m, k), gen).bfloat16()
    q2, s2 = G.mxfp8_e4m3_quantize_shuffled(x2, cfg.tile_m)
    q.copy_(q2)
    s.copy_(s2)
    gr.replay()
    eager = G.mxfp8_gemm_shuffled(q2, s2, w, ws, torch.bfloat16, m)
    torch.cuda.synchronize()
    assert torch.equal(gout, eager)
    # the bf16 entry (the row's own quantizer) gives the same bytes
    assert torch.equal(G.mxfp8_linear(x, w, ws, torch.bfloat16), ref)


def test_no_compile_after_warmup():
    from flydsl.compiler.jit_function import MlirCompiler

    def launch(n, k, m):
        _, _, w, ws = _weights(n, k)
        tile = G.prequant_code_tile_rows(m, n, k)
        q, s = G.mxfp8_e4m3_quantize_shuffled(torch.zeros((m, k), device="cuda", dtype=torch.bfloat16), tile)
        return G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m)

    for n, k, lo, hi, _, _ in ROWS:
        launch(n, k, hi)
    torch.cuda.synchronize()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for n, k, lo, hi, _, _ in ROWS:
            for m in range(lo, hi + 1, 3):
                launch(n, k, m)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig


@pytest.mark.parametrize("n,k,m", [(r[0], r[1], r[3]) for r in ROWS])
def test_mfma_co_residency(n, k, m):
    from tests.mfma_spinner import mfma_spinner

    _, _, w, ws = _weights(n, k)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, G.prequant_code_tile_rows(m, n, k))
    ref = G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m)
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for _ in range(3):
        spinner(100_000, side)
        got = G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m)
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))

