# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
import torch.nn.functional as F
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168 import reference as R
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import (
    indexer_prep,
    supports_indexer_prep,
)

from .cases import EPS, SERVED_M, WEIGHT_SCALE, owned_slots, require_gfx950

TOPK = 2048


def make(n, m, gen, pages, page_stride, scale=2.0):
    mk = lambda *s: (torch.randn(s, generator=gen) * scale).bfloat16().cuda()  # noqa: E731
    q, k, w = mk(n, 32, 128), mk(m, 128), mk(n, 32)
    lw = (torch.randn(128, generator=gen) * 0.3 + 1).bfloat16().cuda()
    lb = (torch.randn(128, generator=gen) * 0.1).bfloat16().cuda()
    base = torch.randint(0, 256, (pages * page_stride,), generator=gen, dtype=torch.uint8).cuda()
    cache = base.as_strided((pages, 64, 68), (page_stride, 68, 1))
    return q, k, w, lw, lb, base, cache


def run(q, k, w, lw, lb, slots, cache, topk_out):
    n = q.shape[0]
    qp = torch.empty((n, 32, 64), dtype=torch.uint8, device="cuda")
    qs = torch.empty((n, 1, 4, 16, 4), dtype=torch.uint8, device="cuda")
    wo = torch.empty((n, 32), dtype=torch.float32, device="cuda")
    indexer_prep(q, k, w, lw, lb, EPS, slots, cache, topk_out, WEIGHT_SCALE, qp, qs, wo)
    return qp, qs, wo


@pytest.mark.parametrize("n", SERVED_M)
@pytest.mark.parametrize("page_stride", [4352, 41984])
@torch.inference_mode()
def test_matches_reference(n, page_stride):
    require_gfx950()
    gen = torch.Generator().manual_seed(n)
    pages = 24
    m = max(1, n - n // 5)  # graph padding: the last query rows have no key
    q, k, w, lw, lb, base, cache = make(n, m, gen, pages, page_stride)
    slots = owned_slots(m, m, pages, gen, owned=3)
    topk_out = torch.randint(-5, 100, (n + 3, TOPK), generator=gen, dtype=torch.int32).cuda()
    before = base.clone().as_strided(cache.shape, cache.stride())
    assert supports_indexer_prep(q, cache, topk_out)
    qp, qs, wo = run(q, k, w, lw, lb, slots, cache, topk_out)
    rp, rs, rw, rt, rc, written = R.indexer_prep_reference(k, lw, lb, EPS, slots, q, w, WEIGHT_SCALE, topk_out, before)
    assert torch.equal(qp, rp) and torch.equal(qs, rs) and torch.equal(wo, rw)
    assert bool((topk_out == -1).all())
    R.check_index(cache, rc, before, written, "cache")
    cache2 = torch.zeros_like(base).as_strided(cache.shape, cache.stride())
    qp2, qs2, wo2 = run(q, k, w, lw, lb, slots, cache2, topk_out)
    assert torch.equal(qp, qp2) and torch.equal(qs, qs2) and torch.equal(wo, wo2)


@pytest.mark.parametrize("scale", [1e-3, 40.0])
@torch.inference_mode()
def test_scale_extremes(scale):
    require_gfx950()
    gen = torch.Generator().manual_seed(5)
    n = 64
    q, k, w, lw, lb, base, cache = make(n, n, gen, 8, 4352, scale)
    slots = torch.arange(n, dtype=torch.int64, device="cuda")
    topk_out = torch.zeros((n, TOPK), dtype=torch.int32, device="cuda")
    before = base.clone().as_strided(cache.shape, cache.stride())
    qp, qs, wo = run(q, k, w, lw, lb, slots, cache, topk_out)
    rp, rs, rw, _, rc, written = R.indexer_prep_reference(k, lw, lb, EPS, slots, q, w, WEIGHT_SCALE, topk_out, before)
    assert torch.equal(qp, rp) and torch.equal(qs, rs) and torch.equal(wo, rw)
    R.check_index(cache, rc, before, written, "cache")


@pytest.mark.parametrize("n", [12, 192])
@torch.inference_mode()
def test_stock_ops(n):
    """layer_norm + Triton store_keys + quantize_queries_and_weights, the ops of the stock chain."""
    require_gfx950()
    fp4 = pytest.importorskip("vllm.models.kimi_k3.amd.ops.fp4_indexer")
    gen = torch.Generator().manual_seed(n)
    pages = 24
    q, k, w, lw, lb, base, cache = make(n, n, gen, pages, 41984)
    slots = owned_slots(n, n, pages, gen, owned=2)
    stock_base = torch.zeros_like(base)
    stock_cache = stock_base.as_strided(cache.shape, cache.stride())
    fp4.store_keys(F.layer_norm(k, (128,), lw, lb, EPS), stock_cache, slots)
    starts = torch.arange(n + 1, device="cuda", dtype=torch.int32)
    sp, ss, sw, _ = fp4.quantize_queries_and_weights(q, w, WEIGHT_SCALE, starts, 0, n)
    ours_base = torch.zeros_like(base)
    ours_cache = ours_base.as_strided(cache.shape, cache.stride())
    topk_out = torch.zeros((n, TOPK), dtype=torch.int32, device="cuda")
    qp, qs, wo = run(q, k, w, lw, lb, slots, ours_cache, topk_out)
    assert torch.equal(qp, sp) and torch.equal(qs, ss) and torch.equal(wo, sw)
    assert (ours_base == stock_base).float().mean() >= R.EXACT_MIN
