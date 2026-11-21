# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168 import reference as R
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import (
    qkv_a_norm_kv_insert,
    supports_qkv_a_norm_kv_insert,
)

from .cases import EPS, SERVED_M, owned_slots, require_gfx950


def make(n, gen, row_pad=0, scale=2.0):
    x = (torch.randn((n, 2112 + row_pad), generator=gen) * scale).bfloat16().cuda()
    qw = (torch.randn(1536, generator=gen) * 0.3 + 1).bfloat16().cuda()
    kw = (torch.randn(512, generator=gen) * 0.3 + 1).bfloat16().cuda()
    return x, qw, kw


def run(x, qw, kw, cache, slots):
    n = x.shape[0]
    q = torch.empty((n, 1536), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((n, 512), dtype=torch.bfloat16, device="cuda")
    qkv_a_norm_kv_insert(x, qw, kw, EPS, q, kv, cache, slots)
    return q, kv


@pytest.mark.parametrize("n", SERVED_M)
@pytest.mark.parametrize("page_stride", [656 * 64, 656 * 64 + 1024])
@torch.inference_mode()
def test_matches_reference(n, page_stride):
    require_gfx950()
    gen = torch.Generator().manual_seed(n)
    pages = 24
    x, qw, kw = make(n, gen)
    base = torch.randint(0, 256, (pages * page_stride,), generator=gen, dtype=torch.uint8).cuda()
    cache = base.as_strided((pages, 64, 656), (page_stride, 656, 1))
    n_slots = max(1, n - n // 5)  # graph padding: the last rows have no slot
    slots = owned_slots(n, n_slots, pages, gen, owned=3)
    before = base.clone().as_strided(cache.shape, cache.stride())
    assert supports_qkv_a_norm_kv_insert(x, cache)
    q, kv = run(x, qw, kw, cache, slots)
    rq, rkv, rc, written = R.qkv_a_norm_kv_insert_reference(x, qw, kw, EPS, before, slots)
    R.check_norm(q, rq, "q")
    R.check_norm(kv, rkv, "kv")
    R.check_kv_cache(cache, rc, before, written, "cache")
    base2 = base.clone()
    base2.fill_(0)
    q2, kv2 = run(x, qw, kw, base2.as_strided(cache.shape, cache.stride()), slots)
    assert torch.equal(q, q2) and torch.equal(kv, kv2)


@torch.inference_mode()
def test_no_cache_and_padded_rows():
    require_gfx950()
    gen = torch.Generator().manual_seed(7)
    x, qw, kw = make(40, gen, row_pad=16)
    xv = x[:, :2112]
    assert not supports_qkv_a_norm_kv_insert(xv[:, ::2], None)
    q, kv = run(x, qw, kw, None, None)
    rq, rkv, _, _ = R.qkv_a_norm_kv_insert_reference(x, qw, kw, EPS)
    R.check_norm(q, rq, "q")
    R.check_norm(kv, rkv, "kv")


@pytest.mark.parametrize("scale", [1e-3, 40.0])
@torch.inference_mode()
def test_scale_extremes(scale):
    require_gfx950()
    gen = torch.Generator().manual_seed(3)
    n, pages = 64, 8
    x, qw, kw = make(n, gen, scale=scale)
    cache = torch.zeros((pages, 64, 656), dtype=torch.uint8, device="cuda")
    slots = torch.arange(n, dtype=torch.int64, device="cuda")
    before = cache.clone()
    q, kv = run(x, qw, kw, cache, slots)
    rq, rkv, rc, written = R.qkv_a_norm_kv_insert_reference(x, qw, kw, EPS, before, slots)
    R.check_norm(q, rq, "q")
    R.check_norm(kv, rkv, "kv")
    R.check_kv_cache(cache, rc, before, written, "cache")


@pytest.mark.parametrize("n", [12, 192])
@torch.inference_mode()
def test_stock_ops(n):
    """AITER rms_norm x2 + concat_and_cache_mla (fp8_ds_mla), the ops of the stock chain."""
    require_gfx950()
    aiter = pytest.importorskip("aiter")
    ops = pytest.importorskip("vllm._custom_ops")
    gen = torch.Generator().manual_seed(n)
    pages = 24
    x, qw, kw = make(n, gen)
    slots = owned_slots(n, n, pages, gen, owned=2)
    stock = torch.zeros((pages, 64, 656), dtype=torch.uint8, device="cuda")
    q_c = aiter.rms_norm(x[:, :1536], qw, EPS)
    kv_c = aiter.rms_norm(x[:, 1536:2048], kw, EPS)
    ops.concat_and_cache_mla(
        kv_c, x[:, 2048:2112], stock, slots, kv_cache_dtype="fp8_ds_mla",
        scale=torch.ones(1, device="cuda"), skip_scale=False,
    )  # fmt: skip
    ours = torch.zeros_like(stock)
    q, kv = run(x, qw, kw, ours, slots)
    R.check_norm(q, q_c, "q")
    R.check_norm(kv, kv_c, "kv")
    own = (slots >= 0).nonzero().flatten()
    pg, off = slots[own] // 64, slots[own] % 64
    assert (ours[pg, off] == stock[pg, off]).float().mean() >= R.EXACT_MIN
    untouched = torch.ones(pages * 64, dtype=torch.bool, device="cuda")
    untouched[slots[own]] = False
    assert not ours.view(-1, 656)[untouched].any()


@pytest.mark.parametrize("n", [1, 12, 65, 192])
@torch.inference_mode()
def test_index_side(n):
    """The row carries the indexer's key and head weights (wk_index / wv_index weights concatenated onto qkv_a):
    key LayerNorm + MXFP4 insert and the weights match ``indexer_prep``'s; the queries-only ``indexer_prep`` completes
    the indexer inputs."""
    from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import indexer_prep
    from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import IndexSide

    from .cases import WEIGHT_SCALE

    require_gfx950()
    gen = torch.Generator().manual_seed(n)
    pages = 24
    x, qw, kw = make(n, gen, row_pad=160)
    lw = (torch.randn(128, generator=gen) * 0.3 + 1).bfloat16().cuda()
    lb = (torch.randn(128, generator=gen) * 0.1).bfloat16().cuda()
    kvc = torch.zeros((pages, 64, 656), dtype=torch.uint8, device="cuda")
    islots = owned_slots(n, n, pages, gen, owned=2)
    mslots = owned_slots(n, n, pages, gen, owned=3)
    icache = torch.zeros((pages, 64, 68), dtype=torch.uint8, device="cuda")
    wts = torch.empty((n, 32), dtype=torch.float32, device="cuda")
    q = torch.empty((n, 1536), dtype=torch.bfloat16, device="cuda")
    kv = torch.empty((n, 512), dtype=torch.bfloat16, device="cuda")
    side = IndexSide(lw, lb, icache, islots, wts, WEIGHT_SCALE)
    qkv_a_norm_kv_insert(x, qw, kw, EPS, q, kv, kvc, mslots, side)
    rq, rkv, rc, written = R.qkv_a_norm_kv_insert_reference(x[:, :2112], qw, kw, EPS, torch.zeros_like(kvc), mslots)
    R.check_norm(q, rq, "q")
    R.check_kv_cache(kvc, rc, torch.zeros_like(kvc), written, "cache")
    iq = (torch.randn((n, 32, 128), generator=gen) * 2).bfloat16().cuda()
    topk = torch.zeros((n, 2048), dtype=torch.int32, device="cuda")
    qp = torch.empty((n, 32, 64), dtype=torch.uint8, device="cuda")
    qs = torch.empty((n, 1, 4, 16, 4), dtype=torch.uint8, device="cuda")
    indexer_prep(iq, None, None, None, None, EPS, None, icache, topk, WEIGHT_SCALE, qp, qs, None)
    zeros = torch.zeros((pages, 64, 68), dtype=torch.uint8, device="cuda")
    rp, rs, _, _, ric, iwritten = R.indexer_prep_reference(
        x[:, 2112:2240].contiguous(), lw, lb, EPS, islots, iq, x[:, 2240:2272].contiguous(), WEIGHT_SCALE, topk, zeros
    )
    assert torch.equal(qp, rp) and torch.equal(qs, rs) and bool((topk == -1).all())
    assert torch.equal(wts, x[:, 2240:2272].float() * WEIGHT_SCALE)
    R.check_index(icache, ric, zeros, iwritten, "index cache")
