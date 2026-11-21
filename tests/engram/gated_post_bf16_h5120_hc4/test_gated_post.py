# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gated_post`` against ``reference.py``: served rows 1-768 with and without a token mask, gates from saturated to
the clamp floor, in-place output, repeat runs bit-identical, graph replay with new inputs, nothing compiling after
the first call's warm-up. And, with vLLM's DeepSeek-V4.1 module present, against its Triton
``_fused_engram_post_wkv_kernel``."""

import pytest
import torch
from amd_kernels.engram import gated_post_bf16_h5120_hc4 as pkg
from amd_kernels.engram.gated_post_bf16_h5120_hc4 import gated_post, gated_post_reference

H, HC = 5120, 4


def require_gfx950():
    from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import available

    if not available():
        pytest.skip("requires gfx950 with FlyDSL")


def inputs(t, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(t, HC, H, generator=g)
    key = hidden.flatten(1).repeat(1, 1)[:, : HC * H] * 0.5 + torch.randn(t, HC * H, generator=g) * scale
    value = torch.randn(t, H, generator=g) * 0.7
    kv = torch.cat([key, value], 1)
    q = torch.randn(HC, H, generator=g) * 0.5 * scale
    k = torch.randn(HC, H, generator=g) * 2.0
    mask = torch.rand(t, generator=g) > 0.2
    return tuple(x.cuda() for x in (hidden.bfloat16(), kv.bfloat16(), q.bfloat16(), k.bfloat16(), mask))


def close(y, r, h, kv):
    """Cosine >= 0.99999; every element within a bf16 ulp of its larger term (a cancelling sum keeps that absolute
    error); under 1e-3 of the elements differ at all."""
    assert y.dtype == r.dtype
    yf, rf = y.float(), r.float()
    cos = torch.nn.functional.cosine_similarity(yf.flatten(), rf.flatten(), dim=0)
    assert cos >= 0.99999, cos
    bound = (h.float().abs() + kv[:, 4 * H :].float().abs().unsqueeze(1)) * 2.0**-7
    assert ((yf - rf).abs() <= bound).all(), (yf - rf).abs().max()
    assert (yf != rf).float().mean() < 1e-3


@pytest.mark.parametrize("t", [1, 7, 16, 100, 384, 768])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("scale", [0.02, 1.0, 8.0])
def test_matches_reference(t, use_mask, scale):
    require_gfx950()
    h, kv, q, k, mask = inputs(t, t + int(scale * 10), scale)
    m = mask if use_mask else None
    y = gated_post(h, kv, q, k, m)
    torch.cuda.synchronize()
    close(y, gated_post_reference(h, kv, q, k, m), h, kv)
    if use_mask:
        assert torch.equal(y[~mask], h[~mask])


def test_in_place_and_repeat():
    require_gfx950()
    h, kv, q, k, mask = inputs(64, 5)
    ref = gated_post(h, kv, q, k, mask)
    h2 = h.clone()
    out = gated_post(h2, kv, q, k, mask, out=h2)
    assert out is h2 and torch.equal(h2, ref)
    assert torch.equal(gated_post(h, kv, q, k, mask), ref)


def test_graph_replay():
    require_gfx950()
    h, kv, q, k, mask = inputs(128, 9)
    gated_post(h, kv, q, k, mask)
    out = torch.empty_like(h)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        gated_post(h, kv, q, k, mask, out=out)
    for seed in (10, 11):
        nh, nkv, _, _, nm = inputs(128, seed)
        h.copy_(nh), kv.copy_(nkv), mask.copy_(nm)
        out.zero_()
        g.replay()
        torch.cuda.synchronize()
        close(out, gated_post_reference(h, kv, q, k, mask), h, kv)


def test_warm_up_covers_both_modes():
    from amd_kernels import warm_up
    from flydsl.compiler.jit_function import MlirCompiler

    require_gfx950()
    h, kv, q, k, mask = inputs(8, 3)
    warm_up.set_auto(True)
    try:
        gated_post(h, kv, q, k, mask)
        gated_post(h, kv, q, k)
        orig = MlirCompiler.compile

        def unexpected(*a, **kw):
            raise AssertionError("compile after warm-up")

        MlirCompiler.compile = unexpected
        try:
            for t in (1, 33, 768):
                x = inputs(t, t)
                gated_post(*x[:4], x[4])
                gated_post(*x[:4])
        finally:
            MlirCompiler.compile = orig
    finally:
        warm_up.set_auto(False)


def test_equals_vllm_triton():
    engram = pytest.importorskip("vllm.models.deepseek_v41.common.engram")
    from vllm.triton_utils import triton

    require_gfx950()
    for t, use_mask in ((16, False), (200, True)):
        h, kv, q, k, mask = inputs(t, 77)
        out = torch.empty_like(h)
        engram._fused_engram_post_wkv_kernel[(t * HC,)](
            h, kv, q, k, mask if use_mask else h, out, t, h.stride(0), h.stride(1), h.stride(2), kv.stride(0),
            kv.stride(1), q.stride(0), q.stride(1), k.stride(0), k.stride(1), mask.stride(0) if use_mask else 0,
            out.stride(0), out.stride(1), out.stride(2), 1e-20, 1e-6, DIM=H, HC_MULT=HC, BLOCK_SIZE=triton.next_power_of_2(H),
            HAS_MASK=use_mask, num_warps=8)
        close(gated_post(h, kv, q, k, mask if use_mask else None), out, h, kv)
