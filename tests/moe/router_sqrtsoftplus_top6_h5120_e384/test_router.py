# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``router_topk`` against its contract (``reference.py``) and the served path (``fp32_router_gemm_dispatch`` +
``topk_hash_softplus_sqrt``) at every padded decode row count and above: the logits (the kernel's own partials summed
in slice order) within fp32 accumulation error of the torch GEMM; the ids and weights equal the reference top-k of those
logits (a row may differ only where the 6th and 7th ``score + bias`` are within 1e-5); ties go to the lowest expert id;
padding rows; logits above the softplus switch; graph replay with poisoned outputs; determinism; nothing compiles after
the warm-up."""

import pytest
import torch
from amd_kernels.moe import router_sqrtsoftplus_top6_h5120_e384 as rt
from amd_kernels.moe.router_sqrtsoftplus_top6_h5120_e384 import (
    logits_reference,
    router_topk,
    score_reference,
    topk_reference,
    warmup_router_topk,
)
from amd_kernels.moe.router_sqrtsoftplus_top6_h5120_e384.host import _tile, shuffle_gate

H, E = 5120, 384
SCALE = 1.5
ROWS = (1, 7, 16, 17, 32, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 2049)
MARGIN = 1e-5


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950 with FlyDSL")


def make(m, seed, wscale=0.02):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, H), generator=g, device="cuda") * torch.exp2(
        torch.randint(-2, 3, (m, 1), generator=g, device="cuda").float()
    )
    w = torch.randn((E, H), generator=g, device="cuda") * wscale
    bias = torch.randn(E, generator=g, device="cuda") * 0.1
    return x.bfloat16(), w.bfloat16(), bias


def run(x, w, bias, **kw):
    m = x.shape[0]
    partials = torch.empty((_tile(m, kw.get('weight_shuffled', False))[2], m, E), dtype=torch.float32, device=x.device)
    out = router_topk(x, w, bias, SCALE, partials=partials, **kw)
    torch.cuda.synchronize()
    logits = partials[0].clone()
    for p in partials[1:]:
        logits = logits + p
    return out, logits


def close_rows(logits, bias):
    """Rows whose 6th and 7th ``score + bias`` are within MARGIN: the ids may legitimately differ there."""
    top = torch.topk(score_reference(logits) + bias, 7, dim=-1)[0]
    return (top[:, 5] - top[:, 6]).abs() < MARGIN * top[:, 5].abs().clamp_min(1.0)


def check(out, logits, bias, is_padding=None):
    tw, ti = out
    rw, ri = topk_reference(logits, bias, SCALE, is_padding)
    ok = ~close_rows(logits, bias)
    assert torch.equal(ti[ok], ri[ok]), "ids"
    assert ((tw[ok] - rw[ok]).abs() <= 2e-6 * rw[ok].abs().clamp_min(1e-3)).all(), "weights"
    live = ok if is_padding is None else ok & ~is_padding
    assert torch.allclose(tw[live].sum(-1), torch.full_like(tw[live].sum(-1), SCALE), rtol=1e-5)


@pytest.mark.parametrize("m", ROWS)
def test_contract(m):
    require_gfx950()
    x, w, bias = make(m, m)
    out, logits = run(x, w, bias)
    ref = logits_reference(x, w)
    assert (logits - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()
    check(out, logits, bias)
    again, _ = run(x, w, bias)
    assert torch.equal(out[0], again[0]) and torch.equal(out[1], again[1])


@pytest.mark.parametrize("m", ROWS)
def test_shuffled_gate_contract(m):
    """The load-time gate layout: same contract, repeat runs bit-identical, and the ids / weights of the plain layout
    wherever the two summation orders agree on the top 7 (a few ulps of the logits)."""
    require_gfx950()
    x, w, bias = make(m, m + 11)
    ws = shuffle_gate(w)
    assert torch.equal(ws.reshape(E // 16, H // 32, 4, 16, 8).permute(0, 3, 1, 2, 4).reshape(E, H), w)  # a permutation
    out, logits = run(x, ws, bias, weight_shuffled=True)
    ref = logits_reference(x, w)
    assert (logits - ref).abs().max().item() <= 1e-5 * ref.abs().max().item()
    check(out, logits, bias)
    again, _ = run(x, ws, bias, weight_shuffled=True)
    assert torch.equal(out[0], again[0]) and torch.equal(out[1], again[1])
    plain, plogits = run(x, w, bias)
    ok = ~close_rows(logits, bias) & ~close_rows(plogits, bias)
    assert torch.equal(out[1][ok], plain[1][ok])


@pytest.mark.parametrize("m", (1, 16, 192, 768))
def test_served_bitwise(m):
    """Ids and weights of the served top-k op on the same logits, bit for bit (the logits are the kernel's own, so the
    GEMM's accumulation order does not enter), at normal and large (above the softplus switch) logits."""
    require_gfx950()
    pytest.importorskip("vllm")
    from vllm import _custom_ops as ops

    for seed, wscale in ((m, 0.02), (m + 1, 0.05), (m + 2, 0.2)):
        x, w, bias = make(m, seed, wscale)
        out, logits = run(x, w, bias)
        bw = torch.empty((m, 6), dtype=torch.float32, device="cuda")
        bi = torch.empty((m, 6), dtype=torch.int32, device="cuda")
        ops.topk_hash_softplus_sqrt(bw, bi, torch.empty_like(bi), logits, True, SCALE, bias, None, None, None, None, 0)
        assert torch.equal(out[1], bi)
        assert torch.equal(out[0], bw)


@pytest.mark.parametrize("m", (16, 192))
def test_served_gemm_and_topk(m):
    """The whole served pair (GEMM kernel and top-k op): equal ids wherever the 6th and 7th ``score + bias`` differ by
    more than 1e-5, weights within 2e-6 relative (the GEMMs sum in different orders)."""
    require_gfx950()
    pytest.importorskip("vllm")
    from vllm import _custom_ops as ops
    import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401  (registers the op)

    x, w, bias = make(m, m + 3)
    out, logits = run(x, w, bias)
    lg = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
    bw = torch.empty((m, 6), dtype=torch.float32, device="cuda")
    bi = torch.empty((m, 6), dtype=torch.int32, device="cuda")
    ops.topk_hash_softplus_sqrt(bw, bi, torch.empty_like(bi), lg, True, SCALE, bias, None, None, None, None, 0)
    ok = ~close_rows(lg, bias)
    assert torch.equal(out[1][ok], bi[ok])
    assert ((out[0][ok] - bw[ok]).abs() <= 2e-6 * bw[ok].abs()).all()


def test_ties_lowest_id():
    require_gfx950()
    m = 16
    x = torch.zeros((m, H), device="cuda", dtype=torch.bfloat16)
    w = torch.zeros((E, H), device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(E, device="cuda")
    for i in (383, 5, 100, 7, 200, 9, 300, 11):
        bias[i] = 1.0
    tw, ti = router_topk(x, w, bias, SCALE)
    assert (ti == torch.tensor([5, 7, 9, 11, 100, 200], device="cuda", dtype=torch.int32)).all()
    assert torch.allclose(tw, torch.full_like(tw, SCALE / 6), rtol=1e-6)
    bias.zero_()
    tw, ti = router_topk(x, w, bias, SCALE)
    assert (ti == torch.arange(6, device="cuda", dtype=torch.int32)).all()


@pytest.mark.parametrize("m", (16, 192))
def test_large_logits_and_padding(m):
    require_gfx950()
    x, w, bias = make(m, 9, wscale=0.2)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    pad[::3] = True
    out, logits = run(x, w, bias, is_padding=pad)
    assert (logits > 20).any()
    check(out, logits, bias, pad)
    assert (out[1][pad] == -1).all() and (out[0][pad] == 0).all()
    assert (out[1][~pad] >= 0).all()


def test_row_stride():
    require_gfx950()
    m = 96
    x, w, bias = make(m, 8)
    wide = torch.zeros((m, H + 64), device="cuda", dtype=torch.bfloat16)
    wide[:, :H] = x
    a, _ = run(wide[:, :H], w, bias)
    b, _ = run(x, w, bias)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_non_finite_logits():
    require_gfx950()
    m = 16
    x, w, bias = make(m, 4)
    x[3, 7] = float("nan")
    x[4, 7] = float("inf")
    (tw, ti), _ = run(x, w, bias)
    assert ((ti >= 0) & (ti < E)).all()
    assert torch.isfinite(tw[[0, 1, 2]]).all()


def test_caller_buffers_and_graph():
    require_gfx950()
    m = 192
    x, w, bias = make(m, 5)
    tw = torch.empty((m, 6), device="cuda")
    ti = torch.empty((m, 6), device="cuda", dtype=torch.int32)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    router_topk(x, w, bias, SCALE, is_padding=pad, topk_weights=tw, topk_ids=ti)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = router_topk(x, w, bias, SCALE, is_padding=pad, topk_weights=tw, topk_ids=ti)
    assert out[0] is tw and out[1] is ti
    for i in range(3):
        x.mul_(0.75).add_(0.0625)
        pad[i] = True
        ew, ei = router_topk(x, w, bias, SCALE, is_padding=pad)
        tw.fill_(float("nan"))
        ti.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(tw, ew) and torch.equal(ti, ei)


def test_warmup_no_compile_after():
    require_gfx950()
    from flydsl.compiler.jit_function import MlirCompiler

    warmup_router_topk()
    for m in ROWS:
        x, w, bias = make(m, 3)
        router_topk(x, w, bias, SCALE)
        router_topk(x, w, bias, SCALE, is_padding=torch.zeros(m, dtype=torch.bool, device="cuda"))
        ws = shuffle_gate(w)
        router_topk(x, ws, bias, SCALE, weight_shuffled=True)
        router_topk(x, ws, bias, SCALE, is_padding=torch.zeros(m, dtype=torch.bool, device="cuda"), weight_shuffled=True)
    torch.cuda.synchronize()
    orig = MlirCompiler.compile

    def unexpected(*a, **kw):
        raise AssertionError("compile after warm-up")

    MlirCompiler.compile = unexpected
    try:
        for m in ROWS:
            x, w, bias = make(m, 3)
            router_topk(x, w, bias, SCALE)
            router_topk(x, w, bias, SCALE, is_padding=torch.zeros(m, dtype=torch.bool, device="cuda"))
            ws = shuffle_gate(w)
            router_topk(x, ws, bias, SCALE, weight_shuffled=True)
            router_topk(x, ws, bias, SCALE, is_padding=torch.zeros(m, dtype=torch.bool, device="cuda"), weight_shuffled=True)
        torch.cuda.synchronize()
    finally:
        MlirCompiler.compile = orig
    assert rt.__name__.endswith("e384")
