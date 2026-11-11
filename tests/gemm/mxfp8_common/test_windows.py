# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode windows of the MXFP8 GEMM table against the contract.

Each window's row, selected by
the table (``gemm._select_config``), against the exact (fp64-accumulated) product of the
same quantized operands rounded once to bf16 (``reference.mxfp8_gemm_reference``), at the
window and tile edges and at M that are not multiples of 16 (the (16, 16) code layout is
k-block-major inside a 16-row tile, so an A bound cut at a row count would drop the K
tail of the last tile's rows; odd RT tiles re-pair the A-scale atoms). Under CUDA graph
replay after the serving warm-up (``warmup_mxfp8_gemm_weight``), replays with changed inputs
give the eager bytes and nothing compiles. The launched kernel names prove the window's
row ran.
"""

import pytest
import torch
from aiter.ops.shuffle import shuffle_weight
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import mxfp8_gemm_reference, quantize_mxfp8

from .test_compile import count_compiles
from .test_gemm import check, require_gfx950

BF16 = torch.bfloat16


def make_operands(n, k, m, seed, zero_row=True):
    torch.manual_seed(seed)
    w = torch.randn(n, k, device="cuda", dtype=BF16) * 0.05
    wc, ws = quantize_mxfp8(w)
    weight, weight_scale = g.preshuffle_mxfp8_weight(wc, ws)
    x = torch.randn(m, k, device="cuda", dtype=BF16) * 0.05
    if zero_row:
        x[:1] = 0.0  # an all-zero row: every block's scale byte clamps to 0
    xq, xs = quantize_mxfp8(x)
    return (wc, ws), (weight, weight_scale), (xq, xs)


def shuffled_activation(codes, exponents, m, rows):
    """The producer's (16, 16) code layout and AITER's shuffled scales for ``m`` rows
    padded to ``rows`` (the row tile ``prequant_code_tile_rows`` reports)."""
    k = codes.shape[1]
    padded = torch.zeros((rows, k), dtype=codes.dtype, device=codes.device)
    padded[:m] = codes
    scale = torch.full((rows, k // 32), 127, dtype=exponents.dtype, device=codes.device)
    scale[:m] = exponents
    return shuffle_weight(padded, layout=(16, 16)).contiguous(), g._preshuffle_activation_scale(scale)


def kernel_names(fn):
    """The device kernel names of one call (a launch-name check, not a timing)."""
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({e.key for e in prof.key_averages() if e.self_device_time_total > 0})


# (window, N, K): the row the table selects inside the window, its kernel-name tag per tile
WINDOWS = {
    "kw4_k8192": (6144, 8192),  # decode 96x96 kw4 sb, M 289..416 (tile 96 / 112)
}
NUMERICS = [
    *[("kw4_k8192", m) for m in (289, 301, 384, 385, 401, 416)],
]
GRAPH = [
    *[("kw4_k8192", m) for m in (289, 384, 416)],
]


def expected(window, m):
    """(kernel, tile_m, a substring of the launched kernel's name)."""
    tile = 96 if m <= 384 else 112
    return "decode", tile, f"_rt{tile // 16}_ni6_kw4_ks1_"


def window_call(window, operands, m):
    """The entry point serving takes for the window's row: the producer's (16, 16) codes
    (``mxfp8_gemm_shuffled``)."""
    (weight, weight_scale), (xq, xs) = operands
    n, k = WINDOWS[window]
    rows = g.prequant_code_tile_rows(m, n, k, BF16)
    a_sh, a_scale = shuffled_activation(xq, xs, m, -(-m // rows) * rows)
    return lambda: g.mxfp8_gemm_shuffled(a_sh, a_scale, weight, weight_scale, BF16, m)


@pytest.mark.parametrize("window,m", NUMERICS, ids=[f"{w}-m{m}" for w, m in NUMERICS])
@torch.inference_mode()
def test_window_matches_reference(window, m):
    require_gfx950()
    n, k = WINDOWS[window]
    name, tile, tag = expected(window, m)
    config = g._select_config(m, n, k, BF16)
    assert (config.kernel, config.tile_m, config.split_k) == (name, tile, 1), config
    assert g.prequant_code_tile_rows(m, n, k, BF16) == tile
    (wc, ws), shuffled_w, (xq, xs) = make_operands(n, k, m, seed=m)
    call = window_call(window, (shuffled_w, (xq, xs)), m)
    names = kernel_names(call)
    assert any(tag in x for x in names), (window, m, names)
    out = call()
    assert out.shape == (m, n)
    ref = mxfp8_gemm_reference(xq, xs, wc, ws, accumulate=torch.float64)
    check(f"{window} M={m}", out, ref)
    assert torch.equal(call().view(torch.uint8), out.view(torch.uint8))  # deterministic


@pytest.mark.parametrize("window,m", GRAPH, ids=[f"{w}-m{m}" for w, m in GRAPH])
@torch.inference_mode()
def test_window_graph_replay(window, m, monkeypatch):
    """After the serving warm-up the captured call compiles nothing, and replays with
    changed inputs give the eager bytes."""
    require_gfx950()
    n, k = WINDOWS[window]
    _, (weight, weight_scale), (xq, xs) = make_operands(n, k, m, seed=m, zero_row=False)
    g._warmup_weight(weight, weight_scale, max(384, m), BF16)
    torch.cuda.synchronize()
    compiled = count_compiles(monkeypatch)
    rows = -(-m // g.prequant_code_tile_rows(m, n, k, BF16)) * g.prequant_code_tile_rows(m, n, k, BF16)
    a_sh, a_scale = shuffled_activation(xq, xs, m, rows)
    call = lambda: g.mxfp8_gemm_shuffled(a_sh, a_scale, weight, weight_scale, BF16, m)  # noqa: E731
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    assert not compiled, f"{window} M={m}: {compiled} compiled after the serving warm-up"
    for seed in (7, 29, 2026):
        torch.manual_seed(seed)
        q, s = quantize_mxfp8(torch.randn(m, k, device="cuda", dtype=BF16) * 0.05)
        sh, sh_s = shuffled_activation(q, s, m, rows)
        a_sh.copy_(sh)
        a_scale.copy_(sh_s)
        graph.replay()
        want = call()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.uint8), want.view(torch.uint8)), seed


# the o_proj_full shape's decode rows (to 416 rows); the TP4 shared
# expert's (rows to 256)
SERVED = [(6144, 8192), (1536, 6144), (6144, 768)]


@pytest.mark.parametrize("n,k", SERVED, ids=[f"n{n}_k{k}" for n, k in SERVED])
@torch.inference_mode()
def test_no_compile_after_serving_warmup(n, k, monkeypatch):
    """Root README rule 4 for the selection: after the warm-up over 416 tokens,
    no M of 1..416 compiles a FlyDSL kernel through any entry point (``mxfp8_linear``,
    ``mxfp8_gemm`` on row-major codes, ``mxfp8_gemm_shuffled`` on a shuffled-A row)."""
    require_gfx950()
    _, (weight, weight_scale), _ = make_operands(n, k, 1, seed=0)
    g._warmup_weight(weight, weight_scale, 416, BF16)
    torch.cuda.synchronize()
    x = torch.randn(416, k, device="cuda", dtype=BF16) * 0.05
    xq, xs = quantize_mxfp8(x)
    compiled = count_compiles(monkeypatch)
    for m in range(1, 417):
        g.mxfp8_linear(x[:m], weight, weight_scale)
        g.mxfp8_gemm(xq[:m], xs[:m], weight, weight_scale, BF16)
        tile = g.prequant_code_tile_rows(m, n, k, BF16)
        if tile:
            a_sh, a_scale = shuffled_activation(xq[:m], xs[:m], m, -(-m // tile) * tile)
            g.mxfp8_gemm_shuffled(a_sh, a_scale, weight, weight_scale, BF16, m)
        assert not compiled, f"N {n} K {k} M {m}: {compiled} compiled after the serving warm-up"
