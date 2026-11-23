# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contexts above the block select's 262144 entries, up to ``MAX_ENTRIES`` = 1048576 (the long path: ``kernel_blocks``,
the candidate consumer's scorer and token select): exact top-512 positions (the fp64 reference, ties to the lower
position, positions ascending, -1 padded) at 524288 and 1048576 entries, the candidate source layer's list and the
consumer layers over it, the ragged lists, rows of every length in one tier (empty, short, below and above the block
select's limit), ties and outliers among the block maxima, equal bits to the block path where both apply (the long path
forced at short contexts), repeat runs, graph replay with new inputs."""

import pytest
import torch
from amd_kernels.indexer import indexer_mxfp4_h32_d128_top512 as ix
from amd_kernels.indexer.indexer_mxfp4_h32_d128_top512 import host

from .test_indexer import DEV, check_rows, ragged_of, require_gfx950

NN = 6


def spread(ctx, reqs, lo=0.55):
    out = [int(ctx * (lo + (1 - lo) * (i + 1) / reqs)) for i in range(reqs)]
    out[-1] = ctx
    return out


def run(inputs, nn, ctx, *, cand=False, ragged=None):
    q, qs, w, cache, bt, ends = inputs
    out = torch.full((q.shape[0], 512), -9, dtype=torch.int32, device=DEV)
    c = torch.full((q.shape[0], 2048), -9, dtype=torch.int32, device=DEV) if cand else None
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, candidates_out=c, ragged=ragged)
    torch.cuda.synchronize()
    return out, c


def check_candidates(cand_row, x):
    """The source layer's list of a row: ascending, -1 padded, the newest key's block and the 2047 best others (up to
    rounding at the last rank)."""
    scores = ix.block_scores_reference(x)
    n = scores.numel()
    got = cand_row[cand_row >= 0].long()
    assert got.numel() == min(2048, n) and bool((cand_row[got.numel():] == -1).all()), (got.numel(), n)
    assert bool((got[1:] > got[:-1]).all())
    if n == 0:
        return
    assert bool((got == n - 1).any()), "the newest key's block"
    rest = got[got != n - 1]
    others = torch.cat([scores[: n - 1], scores[n:]])
    compact = torch.full((2048,), -1, dtype=torch.int32, device=DEV)
    compact[: rest.numel()] = rest.int()
    if n > 1:
        assert ix.topk_matches(compact, others, 2047) is None


@pytest.mark.parametrize("ctx,reqs,P", [(524288, 3, 128), (524288, 4, 64), (1048576, 3, 128), (1048576, 1, 128)])
def test_long_rows_are_exact(ctx, reqs, P):
    require_gfx950()
    inputs = ix.random_inputs(spread(ctx, reqs), NN, P, DEV, seed=ctx + reqs)
    out, cand = run(inputs, NN, ctx, cand=True)
    plain, _ = run(inputs, NN, ctx)
    assert torch.equal(out, plain)  # the source layer's top-512 is the dense layer's
    again, _ = run(inputs, NN, ctx)
    assert torch.equal(again, out)
    logits = check_rows(out, inputs, NN, P, equal_frac=0.9)
    for r, x in enumerate(logits):
        check_candidates(cand[r], x)


@pytest.mark.parametrize("ctx", [524288, 1048576])
def test_candidate_consumer_over_long_rows(ctx):
    """The consumer layers score the source layer's 2048 candidate blocks with their own q and weights (positions up to
    1M), and write the ragged lists."""
    require_gfx950()
    reqs = 3
    inputs = ix.random_inputs(spread(ctx, reqs), NN, 128, DEV, seed=ctx)
    q, qs, w, cache, bt, ends = inputs
    rows = q.shape[0]
    _, cand = run(inputs, NN, ctx, cand=True)
    g = torch.Generator(device=DEV).manual_seed(13)
    q2 = torch.randint(0, 256, q.shape, dtype=torch.uint8, device=DEV, generator=g)
    w2 = torch.randn_like(w) * 0.1
    cand[::4, 300:] = -1  # a short pool on some rows
    rg = ragged_of(rows, reqs, ctx, 128, 3)
    out = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_candidate_topk(q2, qs, w2, cache, bt, ends, cand, out, next_n=NN, max_ctx=ctx, ragged=rg)
    torch.cuda.synchronize()
    for r, x in enumerate(ix.logits_reference(q2, qs, w2, cache, bt, ends, NN, 128)):
        got = out[r]
        sel = got[got >= 0].long()
        ids = cand[r][cand[r] >= 0].long()
        pos = (ids[:, None] * 8 + torch.arange(8, device=DEV)).flatten()
        pos = pos[pos < x.numel()]
        at = torch.searchsorted(pos, sel)
        assert bool((pos[at.clamp(max=pos.numel() - 1)] == sel).all()), r  # chosen inside the pool
        compact = torch.full((512,), -1, dtype=torch.int32, device=DEV)
        compact[: at.numel()] = at.int()
        assert ix.topk_matches(compact, x[pos]) is None, r
    slots, indptr, lens_ = ix.ragged_reference(out, NN, rg.slot_block_table, 128, rg.is_valid)
    assert torch.equal(rg.lens, lens_) and torch.equal(rg.indptr, indptr)
    n = int(indptr[-1])
    assert torch.equal(rg.slots[:n], slots[:n])


@pytest.mark.parametrize("bs", [64, 128, 256])
def test_rows_of_every_length_in_the_long_tier(bs):
    """One call, tier 1M: empty, tiny and short rows, rows on both sides of the block select's 262144 entries, the
    longest: every row exact, with ragged lists (the rows of both paths share one list)."""
    require_gfx950()
    ctx = 1048576
    lens = [0, 1, 5, 100, 4097, 100000, 262143, 262150, 300000, 600000, ctx]
    inputs = ix.random_inputs(lens, NN, 128, DEV, seed=77)
    rows = inputs[0].shape[0]
    rg = ragged_of(rows, len(lens), ctx, bs, 5)
    out, cand = run(inputs, NN, ctx, cand=True, ragged=rg)
    check_rows(out, inputs, NN, 128, equal_frac=0.9)
    for r, x in enumerate(ix.logits_reference(*inputs[:5], inputs[5], NN, 128)):
        check_candidates(cand[r], x)
    slots, indptr, lens_ = ix.ragged_reference(out, NN, rg.slot_block_table, bs, rg.is_valid)
    assert torch.equal(rg.lens, lens_) and torch.equal(rg.indptr, indptr)
    n = int(indptr[-1])
    assert torch.equal(rg.slots[:n], slots[:n])


@pytest.mark.parametrize("ties", ["none", "equal_pages", "zero_q", "outliers"])
def test_ties_and_outliers_among_block_maxima(ties):
    """Equal block maxima (one page repeated; zero queries: every logit 0) cut by block id, and a few keys whose scales
    put their logits ~2^40 above the rest (the buckets of the k-th key must not be crowded by the outliers)."""
    require_gfx950()
    ctx = 524288
    lens = spread(ctx, 2)
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, NN, 128, DEV, seed=91)
    if ties == "equal_pages":
        cache[:] = cache[0:1]
    elif ties == "zero_q":
        q.zero_()
    elif ties == "outliers":
        flat = cache.view(cache.shape[0], -1)
        flat[:40:3, 128 * 64:] = 167  # the e8m0 scales of a few pages: their logits are 2^40 times the rest
    out, cand = run((q, qs, w, cache, bt, ends), NN, ctx, cand=True)
    logits = ix.logits_reference(q, qs, w, cache, bt, ends, NN, 128)
    if ties == "zero_q":
        first = torch.arange(512, dtype=torch.int32, device=DEV)
        assert all(torch.equal(out[r], first) for r in range(out.shape[0]))
    for r, x in enumerate(logits):
        if ties in ("equal_pages", "zero_q"):
            assert torch.equal(out[r], ix.select_reference(x)), r
        else:
            assert ix.topk_matches(out[r], x) is None, r
        if ties == "outliers":
            check_candidates(cand[r], x)
        else:
            assert torch.equal(cand[r], ix.candidates_reference(x)), r


@pytest.mark.parametrize(
    "lens,nn,P,ties",
    [
        ([90000, 70000, 40000, 33000, 32768, 20000, 5000, 100, 0, 1], 6, 128, "none"),
        ([100000, 64000, 31000, 12000], 3, 64, "none"),
        ([100000, 48000, 9000, 40], 6, 128, "equal_pages"),
        ([100000, 48000, 9000, 40], 6, 64, "zero_q"),
        ([140000, 100000, 66000, 3000, 12], 6, 128, "none"),  # two segments
        ([140000, 70000, 9000], 6, 64, "equal_pages"),
        ([66000, 50000], 8, 128, "none"),
        ([70000, 4000], 1, 64, "none"),
    ],
)
def test_long_path_equals_block_path(monkeypatch, lens, nn, P, ties):
    """The long path forced at contexts the block path serves: positions, candidate lists and ragged lists are bit for
    bit the block path's (same arithmetic, same tie rule), also for rows below the forced limit (served by the block
    select in the same call)."""
    require_gfx950()
    ctx = max(lens)
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=ctx + nn)
    if ties == "equal_pages":
        cache[:] = cache[0:1]
    elif ties == "zero_q":
        q[::2] = 0
    rows = q.shape[0]
    inputs = (q, qs, w, cache, bt, ends)
    rg_b, rg_l = ragged_of(rows, len(lens), ctx, 128, 3), ragged_of(rows, len(lens), ctx, 128, 3)
    out_b, cand_b = run(inputs, nn, ctx, cand=True)
    out_b2, _ = run(inputs, nn, ctx, ragged=rg_b)
    cons_b = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_candidate_topk(q, qs, w, cache, bt, ends, cand_b, cons_b, next_n=nn, max_ctx=ctx)
    monkeypatch.setattr(host, "BLOCK_CTX_MAX", 16384)
    assert host.blocks_tier(host.BLOCK_CTX_MAX) * 8 < ctx
    out_l, cand_l = run(inputs, nn, ctx, cand=True)
    out_l2, _ = run(inputs, nn, ctx, ragged=rg_l)
    cons_l = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_candidate_topk(q, qs, w, cache, bt, ends, cand_l, cons_l, next_n=nn, max_ctx=ctx)
    torch.cuda.synchronize()
    assert torch.equal(out_l, out_b) and torch.equal(cand_l, cand_b)
    assert torch.equal(out_l2, out_b2) and torch.equal(cons_l, cons_b)
    for f in ("slots", "indptr", "lens"):
        assert torch.equal(getattr(rg_l, f), getattr(rg_b, f)), f


def test_graph_replay_long_with_new_inputs():
    require_gfx950()
    ctx = 1048576
    lens = [300000, 40000, 700000, 120000, 20000, 1048576]
    nn, P = 6, 128
    inputs = ix.random_inputs(lens, nn, P, DEV, seed=21)
    q, qs, w, cache, bt, ends = inputs
    out = torch.full((q.shape[0], 512), -9, dtype=torch.int32, device=DEV)
    cand = torch.full((q.shape[0], 2048), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, candidates_out=cand)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, candidates_out=cand)
    for seed in (22, 23):
        b = ix.random_inputs(lens, nn, P, DEV, seed=seed, extra_pages=cache.shape[0] - sum(-(-c // P) for c in lens))
        cache.copy_(b[3])
        for dst, src in zip((q, qs, w, bt, ends), (b[0], b[1], b[2], b[4], b[5])):
            dst.copy_(src)
        out.fill_(-9)
        g.replay()
        torch.cuda.synchronize()
        eager, eager_c = run((q, qs, w, cache, bt, ends), nn, ctx, cand=True)
        assert torch.equal(out, eager) and torch.equal(cand, eager_c)
        check_rows(out, (q, qs, w, cache, bt, ends), nn, P, equal_frac=0.9)


def test_max_entries_is_the_limit():
    assert ix.MAX_ENTRIES == 1 << 20
    if not ix.available():
        pytest.skip("requires gfx950 with FlyDSL")
    assert ix.blocker(128, ix.MAX_ENTRIES, 6) is None
    assert "1048577" in ix.blocker(128, ix.MAX_ENTRIES + 1, 6)
