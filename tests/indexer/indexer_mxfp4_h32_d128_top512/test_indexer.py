# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP4 indexer against ``reference.py``: block maxima (cosine >= 0.99999 of the exact fp64 ones), the top-512
positions (the exact selection up to rounding at the 512th rank, equal to the reference when no logits are close), the
candidate consumer, rows per request 1-8, 64- and 128-entry pages, contexts 4k-130k at 3-128 requests (decode rows
18-768),
short and empty rows, tied logits (equal keys, zero queries), repeat runs bit-identical, graph replay with new inputs;
the first call's warm-up compiles every variant and nothing compiles later (the long path of ``test_long_context.py``
included)."""

import json
import os
import subprocess
import sys

import pytest
import torch
from amd_kernels.indexer import indexer_mxfp4_h32_d128_top512 as ix

DEV = "cuda"


def require_gfx950():
    if not ix.available():
        pytest.skip("requires gfx950 with FlyDSL")


def lengths_of(ctx: int, reqs: int, lo: float = 0.5):
    return [int(ctx * (lo + (1 - lo) * (i + 0.5) / reqs)) for i in range(reqs)]


def run_dense(inputs, nn, P, ctx):
    q, qs, w, cache, bt, ends = inputs
    out = torch.full((q.shape[0], ix.reference.TOPK), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx)
    return out


def check_rows(out, inputs, nn, P, equal_frac=0.9):
    q, qs, w, cache, bt, ends = inputs
    logits = ix.logits_reference(q, qs, w, cache, bt, ends, nn, P)
    exact = 0
    for r, x in enumerate(logits):
        msg = ix.topk_matches(out[r], x)
        assert msg is None, (r, int(ends[r]), msg)
        exact += int(bool((out[r] == ix.select_reference(x)).all()))
    assert exact >= equal_frac * len(logits), (exact, len(logits))
    return logits


@pytest.mark.parametrize("reqs", [3, 11, 32, 64, 128])
@pytest.mark.parametrize("ctx", [4096, 16384, 65536, 131072])
def test_dense_matches_reference(reqs, ctx):
    require_gfx950()
    P = 64 if ctx <= 65536 else 128
    inputs = ix.random_inputs(lengths_of(ctx, reqs), 6, P, DEV, seed=reqs + ctx)
    out = run_dense(inputs, 6, P, ctx)
    torch.cuda.synchronize()
    check_rows(out, inputs, 6, P)


@pytest.mark.parametrize("nn", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("P", [64, 128])
def test_rows_per_request_and_pages(nn, P):
    require_gfx950()
    lens = [9000, 513, 4100, 17000, 300]
    inputs = ix.random_inputs(lens, nn, P, DEV, seed=nn * 7 + P)
    check_rows(run_dense(inputs, nn, P, 17000), inputs, nn, P)


@pytest.mark.parametrize("reqs,ctx,P", [(4, 20000, 128), (7, 60000, 64)])
def test_reference_and_selection_match_aiter(reqs, ctx, P):
    """The contract is AITER's: its logits kernel agrees with ``logits_reference`` (cosine, max error) and its top-512
    chooses the same keys as this one up to keys within rounding of the 512th logit."""
    require_gfx950()
    try:
        from aiter.ops.topk import top_k_per_row_decode
        from aiter.ops.triton.attention import pa_mqa_logits_mxfp4 as pa
    except ImportError:
        pytest.skip("requires AITER")
    nn = 6
    q, qs, w, cache, bt, ends = ix.random_inputs(lengths_of(ctx, reqs), nn, P, DEV, seed=17)
    rows = q.shape[0]
    cl = (ends.view(reqs, nn)[:, -1]).contiguous()
    width = -(-ctx // 128) * 128
    logits = torch.empty(rows, width, dtype=torch.float32, device=DEV)
    pa.paged_mxfp4_mqa_logits(
        q.view(reqs, nn, 32, 64), qs.view(reqs, nn, 32, 4), cache.view(-1, P, 1, 68), w, cl, bt, width,
        out_logits=logits, clean_logits=False, row_ends=ends,
    )  # fmt: skip
    theirs = torch.empty(rows, 512, dtype=torch.int32, device=DEV)
    top_k_per_row_decode(logits, 1, ends, theirs, rows, logits.stride(0), logits.stride(1), k=512)
    mine = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, mine, next_n=nn, max_ctx=ctx)
    torch.cuda.synchronize()
    for r, x in enumerate(ix.logits_reference(q, qs, w, cache, bt, ends, nn, P)):
        e = int(ends[r])
        got = logits[r, :e].double()
        assert float(torch.nn.functional.cosine_similarity(got, x, dim=0)) >= 0.99999
        assert float((got - x).abs().max()) <= 1e-5 * float(x.abs().max()), r
        other = torch.sort(theirs[r][theirs[r] >= 0]).values.int()
        compact = torch.full((512,), -1, dtype=torch.int32, device=DEV)
        compact[: other.numel()] = other
        assert ix.topk_matches(compact, x) is None, r  # AITER's own choice is a valid top-512 of the contract
        assert ix.topk_matches(mine[r], x) is None, r


def test_block_scores():
    require_gfx950()
    P, nn = 128, 6
    lens = [2000, 40000, 777]
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=3)
    scores = ix.indexer_block_scores(q, qs, w, cache, bt, ends, next_n=nn, max_ctx=40000)
    torch.cuda.synchronize()
    for r, x in enumerate(ix.logits_reference(q, qs, w, cache, bt, ends, nn, P)):
        ref = ix.block_scores_reference(x)
        got = scores[r, : ref.numel()].double()
        cos = torch.nn.functional.cosine_similarity(got, ref, dim=0)
        assert float(cos) >= 0.99999 and float((got - ref).abs().max()) <= 1e-5 * float(ref.abs().max()), r


def test_short_and_empty_rows():
    require_gfx950()
    P, nn = 128, 6
    lens = [0, 1, 5, 6, 7, 100, 511, 512, 513, 1023, 4097]
    inputs = ix.random_inputs(lens, nn, P, DEV, seed=5)
    check_rows(run_dense(inputs, nn, P, 4100), inputs, nn, P)


def test_ties_are_broken_by_position():
    require_gfx950()
    P, nn = 128, 6
    lens = [30000, 3000]
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=9)
    cache[:] = cache[0:1]  # every key is the same entry sequence of page 0: equal logits at equal in-page offsets
    out = run_dense((q, qs, w, cache, bt, ends), nn, P, 30000)
    q0 = q.clone()
    q0.zero_()  # all logits zero: the first 512 positions
    out0 = run_dense((q0, qs, w, cache, bt, ends), nn, P, 30000)
    torch.cuda.synchronize()
    logits = ix.logits_reference(q, qs, w, cache, bt, ends, nn, P)
    for r, x in enumerate(logits):
        assert torch.equal(out[r], ix.select_reference(x)), r
    first = torch.arange(512, dtype=torch.int32, device=DEV)
    assert all(torch.equal(out0[r], first) for r in range(out0.shape[0]))


def test_repeat_runs_are_bit_identical():
    require_gfx950()
    inputs = ix.random_inputs(lengths_of(65536, 16), 6, 64, DEV, seed=11)
    a, b = run_dense(inputs, 6, 64, 65536), run_dense(inputs, 6, 64, 65536)
    torch.cuda.synchronize()
    assert torch.equal(a, b)


@pytest.mark.parametrize("reqs,ctx,P", [(11, 131072, 128), (32, 16384, 128), (64, 65536, 64)])
def test_candidate_consumer(reqs, ctx, P):
    require_gfx950()
    nn = 6
    q, qs, w, cache, bt, ends = ix.random_inputs(lengths_of(ctx, reqs), nn, P, DEV, seed=reqs)
    logits = ix.logits_reference(q, qs, w, cache, bt, ends, nn, P)
    cand = torch.stack([ix.candidates_reference(x) for x in logits]).contiguous()
    cand[::3, 100:] = -1  # a short pool
    out = torch.full((q.shape[0], 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_candidate_topk(q, qs, w, cache, bt, ends, cand, out, next_n=nn, max_ctx=ctx)
    torch.cuda.synchronize()
    for r, x in enumerate(logits):
        ids = cand[r][cand[r] >= 0].long()
        pos = (ids[:, None] * 8 + torch.arange(8, device=DEV)).flatten()
        pos = pos[pos < x.numel()]
        got = out[r]
        sel = got[got >= 0].long()
        at = torch.searchsorted(pos, sel)
        assert bool((pos[at.clamp(max=pos.numel() - 1)] == sel).all()), r  # chosen inside the pool
        compact = torch.full((512,), -1, dtype=torch.int32, device=DEV)
        compact[: at.numel()] = at.int()
        assert ix.topk_matches(compact, x[pos]) is None, r


@pytest.mark.parametrize("reqs,ctx,P", [(5, 131072, 128), (3, 9000, 128), (4, 1500, 64), (6, 40000, 64)])
def test_candidate_source(reqs, ctx, P):
    require_gfx950()
    nn = 6
    lens = lengths_of(ctx, reqs) + [3, 0]
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=ctx)
    out = torch.full((q.shape[0], 512), -9, dtype=torch.int32, device=DEV)
    cand = torch.full((q.shape[0], 2048), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, candidates_out=cand)
    torch.cuda.synchronize()
    same = 0
    for r, x in enumerate(ix.logits_reference(q, qs, w, cache, bt, ends, nn, P)):
        assert ix.topk_matches(out[r], x) is None, r
        scores = ix.block_scores_reference(x)
        n = scores.numel()
        got = cand[r][cand[r] >= 0].long()
        assert bool((cand[r][got.numel() :] == -1).all()) and bool((got[1:] > got[:-1]).all()), r
        assert got.numel() == min(2048, n), (r, got.numel(), n)
        same += int(torch.equal(cand[r], ix.candidates_reference(x)))
        if n == 0:
            continue
        assert int(got[-1]) == n - 1 or bool((got == n - 1).any()), r  # the newest key's block
        rest = got[got != n - 1]
        others = torch.cat([scores[: n - 1], scores[n:]])
        compact = torch.full((2048,), -1, dtype=torch.int32, device=DEV)
        compact[: rest.numel()] = rest.int()
        if n > 1:
            assert ix.topk_matches(compact, others, 2047) is None, r
    assert same >= 0.9 * q.shape[0]


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("bs", [64, 128])
def test_ragged_lists(candidate, bs):
    """The select's ragged slot lists are ``ragged_reference`` of its own top-512 positions."""
    require_gfx950()
    P, nn, ctx = 128, 6, 30000
    lens = lengths_of(ctx, 9) + [200, 0]
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=31)
    rows = q.shape[0]
    g = torch.Generator(device=DEV).manual_seed(5)
    slot_bt = torch.randint(0, 5000, (len(lens), -(-ctx // bs) + 3), dtype=torch.int32, device=DEV, generator=g)
    valid = (torch.arange(rows, device=DEV) % 7 != 3).int()
    out = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    rg = ix.Ragged(
        torch.full((rows * 512,), -9, dtype=torch.int32, device=DEV),
        torch.full((rows + 1,), -9, dtype=torch.int32, device=DEV),
        torch.full((rows,), -9, dtype=torch.int32, device=DEV),
        valid,
        slot_bt,
        bs,
    )
    if candidate:
        cand = torch.full((rows, 2048), -9, dtype=torch.int32, device=DEV)
        ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, candidates_out=cand)
        torch.cuda.synchronize()
        ix.indexer_candidate_topk(q, qs, w, cache, bt, ends, cand, out, next_n=nn, max_ctx=ctx, ragged=rg)
    else:
        ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, ragged=rg)
    torch.cuda.synchronize()
    slots, indptr, lens_ = ix.ragged_reference(out, nn, slot_bt, bs, valid)
    assert torch.equal(rg.lens, lens_) and torch.equal(rg.indptr, indptr)
    n = int(indptr[-1])
    assert torch.equal(rg.slots[:n], slots[:n])


def test_graph_replay_with_new_inputs():
    require_gfx950()
    P, nn, ctx = 128, 6, 20000
    lens = lengths_of(ctx, 12)
    a = ix.random_inputs(lens, nn, P, DEV, seed=21)
    q, qs, w, cache, bt, ends = a
    out = torch.full((q.shape[0], 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx)
    for seed in (22, 23):
        b = ix.random_inputs(lens, nn, P, DEV, seed=seed, extra_pages=cache.shape[0] - sum(-(-c // P) for c in lens))
        cache.copy_(b[3])
        for dst, src in zip((q, qs, w, bt, ends), (b[0], b[1], b[2], b[4], b[5])):
            dst.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        check_rows(out, (q, qs, w, cache, bt, ends), nn, P)



def block_path(monkeypatch):
    """Force the block-maximum select (no token-logit path) to compare the two."""
    from amd_kernels.indexer.indexer_mxfp4_h32_d128_top512 import host

    monkeypatch.setattr(host, "tokens_tier", lambda max_ctx: None)


def ragged_of(rows, nreq, ctx, bs, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    slot_bt = torch.randint(0, 5000, (nreq, -(-ctx // bs) + 3), dtype=torch.int32, device=DEV, generator=g)
    valid = (torch.arange(rows, device=DEV) % 5 != 2).int()
    return ix.Ragged(
        torch.full((rows * 512,), -9, dtype=torch.int32, device=DEV),
        torch.full((rows + 1,), -9, dtype=torch.int32, device=DEV),
        torch.full((rows,), -9, dtype=torch.int32, device=DEV),
        valid,
        slot_bt,
        bs,
    )


@pytest.mark.parametrize(
    "ctx,lens,nn,P",
    [
        (4096, [4096, 3000, 513, 100, 1, 0, 4000], 6, 64),
        (4097, [4097, 600, 4000], 6, 64),  # the 8192 tier
        (8192, [8192, 100, 4300, 7000, 512, 511], 3, 128),
        (8193, [8193, 3000, 5555], 6, 128),  # the 16384 tier
        (16384, [16384, 9000, 5000, 8193, 4097, 700, 12345], 6, 64),
        (16384, [16000, 16384, 12000, 3000, 20, 7777, 9999, 15000], 8, 128),
        (4096, [4096, 2048, 2], 1, 64),
    ],
)
@pytest.mark.parametrize("ties", [False, True])
def test_token_path_equals_block_path(monkeypatch, ctx, lens, nn, P, ties):
    """Short contexts select from the walk's token logits (kernel_tokens); the logits are the same bits as the block
    path's
    rescoring, so its positions and ragged slot lists equal the block path's, ties (equal logits: one page repeated,
    zero queries) included."""
    require_gfx950()
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=ctx + nn + len(lens))
    if ties:
        cache[:] = cache[0:1]
        q[::2] = 0
    rows, bs = q.shape[0], 128
    rg_t, rg_b = ragged_of(rows, len(lens), ctx, bs, 3), ragged_of(rows, len(lens), ctx, bs, 3)
    out_t = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    out_b = out_t.clone()
    ix.indexer_topk(q, qs, w, cache, bt, ends, out_t, next_n=nn, max_ctx=ctx, ragged=rg_t)
    block_path(monkeypatch)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out_b, next_n=nn, max_ctx=ctx, ragged=rg_b)
    torch.cuda.synchronize()
    assert torch.equal(out_t, out_b)
    for f in ("slots", "indptr", "lens"):
        assert torch.equal(getattr(rg_t, f), getattr(rg_b, f)), f
    check_rows(out_t, (q, qs, w, cache, bt, ends), nn, P, equal_frac=0.0)



@pytest.mark.parametrize("ctx,P", [(4096, 64), (10000, 128), (16384, 64)])
def test_token_path_candidate_source(monkeypatch, ctx, P):
    """The source layer at short contexts: a row of at most 2048 blocks has every block as a candidate; the token path's
    list and positions equal the block path's."""
    require_gfx950()
    nn = 6
    lens = lengths_of(ctx, 5, 0.3) + [7, 0]
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=ctx)
    rows = q.shape[0]
    out_t = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    cand_t = torch.full((rows, 2048), -9, dtype=torch.int32, device=DEV)
    out_b, cand_b = out_t.clone(), cand_t.clone()
    ix.indexer_topk(q, qs, w, cache, bt, ends, out_t, next_n=nn, max_ctx=ctx, candidates_out=cand_t)
    block_path(monkeypatch)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out_b, next_n=nn, max_ctx=ctx, candidates_out=cand_b)
    torch.cuda.synchronize()
    assert torch.equal(out_t, out_b) and torch.equal(cand_t, cand_b)



@pytest.mark.parametrize(
    "ctx,lens,nn,P",
    [
        (131072, [131072, 90000, 40000, 20000], 6, 128),
        (65536, [65536, 30000, 17000, 9000, 2000, 7, 0], 6, 64),
        (20000, [20000, 19000, 100], 3, 128),
        (131072, [131000, 120000], 8, 128),
    ],
)
def test_candidate_consumer_split_equals_gather_select(monkeypatch, ctx, lens, nn, P):
    """A few rows: the candidates are scored by (row, segment) workgroups and the token select picks the 512 best; the
    positions and ragged lists equal the one-workgroup-a-row gather select's (the scoring is the same arithmetic)."""
    require_gfx950()
    from amd_kernels.indexer.indexer_mxfp4_h32_d128_top512 import host

    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=ctx + nn)
    rows = q.shape[0]
    cand = torch.full((rows, 2048), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, torch.empty(rows, 512, dtype=torch.int32, device=DEV), next_n=nn,
                    max_ctx=ctx, candidates_out=cand)  # fmt: skip
    cand[::3, 300:] = -1  # a short pool on some rows
    rg_s, rg_f = ragged_of(rows, len(lens), ctx, 128, 3), ragged_of(rows, len(lens), ctx, 128, 3)
    out_s = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    out_f = out_s.clone()
    ix.indexer_candidate_topk(q, qs, w, cache, bt, ends, cand, out_s, next_n=nn, max_ctx=ctx, ragged=rg_s)
    monkeypatch.setattr(host, "CAND_SPLIT_ROWS", 0)
    ix.indexer_candidate_topk(q, qs, w, cache, bt, ends, cand, out_f, next_n=nn, max_ctx=ctx, ragged=rg_f)
    torch.cuda.synchronize()
    assert torch.equal(out_s, out_f)
    for f in ("slots", "indptr", "lens"):
        assert torch.equal(getattr(rg_s, f), getattr(rg_f, f)), f


def test_token_path_ragged_lists_and_repeat_runs():
    require_gfx950()
    P, nn, ctx, bs = 64, 6, 12000, 128
    lens = lengths_of(ctx, 9) + [200, 0]
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=41)
    rows = q.shape[0]
    rg = ragged_of(rows, len(lens), ctx, bs, 7)
    out = torch.full((rows, 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx, ragged=rg)
    again = out.clone()
    ix.indexer_topk(q, qs, w, cache, bt, ends, again, next_n=nn, max_ctx=ctx, ragged=rg)
    torch.cuda.synchronize()
    assert torch.equal(out, again)
    slots, indptr, lens_ = ix.ragged_reference(out, nn, rg.slot_block_table, bs, rg.is_valid)
    assert torch.equal(rg.lens, lens_) and torch.equal(rg.indptr, indptr)
    n = int(indptr[-1])
    assert torch.equal(rg.slots[:n], slots[:n])


def test_token_path_graph_replay_with_new_inputs():
    require_gfx950()
    P, nn, ctx = 64, 6, 7000
    lens = lengths_of(ctx, 12)
    q, qs, w, cache, bt, ends = ix.random_inputs(lens, nn, P, DEV, seed=51)
    out = torch.full((q.shape[0], 512), -9, dtype=torch.int32, device=DEV)
    ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        ix.indexer_topk(q, qs, w, cache, bt, ends, out, next_n=nn, max_ctx=ctx)
    for seed in (52, 53):
        b = ix.random_inputs(lens, nn, P, DEV, seed=seed, extra_pages=cache.shape[0] - sum(-(-c // P) for c in lens))
        cache.copy_(b[3])
        for dst, src in zip((q, qs, w, bt, ends), (b[0], b[1], b[2], b[4], b[5])):
            dst.copy_(src)
        g.replay()
        torch.cuda.synchronize()
        check_rows(out, (q, qs, w, cache, bt, ends), nn, P)


def scenario():
    from flydsl.compiler.jit_function import MlirCompiler

    from amd_kernels import warm_up

    warm_up.set_auto(True)
    late, original = [], MlirCompiler.compile

    def spy(*a, **k):
        if warm_up._current.get() is not None and not warm_up._in_warm_up.get():
            late.append(1)
        return original(*a, **k)

    MlirCompiler.compile = spy
    bad = []
    for nn, P, ctx in ((6, 128, 100000), (6, 64, 50000), (6, 128, 10000), (4, 128, 20000), (6, 128, 1048576)):
        before = len(late)
        inputs = ix.random_inputs(lengths_of(ctx, 3), nn, P, DEV)
        out = run_dense(inputs, nn, P, ctx)
        rows = out.shape[0]
        cand = torch.full((rows, 2048), -1, dtype=torch.int32, device=DEV)
        sbt = torch.zeros(3, ctx // 64 + 2, dtype=torch.int32, device=DEV)
        rg = ix.Ragged(
            torch.empty(rows * 512, dtype=torch.int32, device=DEV),
            torch.empty(rows + 1, dtype=torch.int32, device=DEV),
            torch.empty(rows, dtype=torch.int32, device=DEV),
            torch.ones(rows, dtype=torch.int32, device=DEV),
            sbt,
            64,
        )
        ix.indexer_topk(*inputs[:5], inputs[5], out, next_n=nn, max_ctx=ctx, candidates_out=cand, ragged=rg)
        ix.indexer_candidate_topk(*inputs[:5], inputs[5], cand, out, next_n=nn, max_ctx=ctx, ragged=rg)
        if len(late) > before:
            bad.append((nn, P, ctx))
    torch.cuda.synchronize()
    return {"late": bad, "reported": warm_up.late_compiles(), "warmed": len(warm_up.warmed())}


def test_first_call_warm_up_compiles_every_variant(tmp_path):
    require_gfx950()
    proc = subprocess.run(
        [sys.executable, "-m", __name__],
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "FLYDSL_RUNTIME_CACHE_DIR": str(tmp_path / "flydsl")},
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["late"] == [] and result["reported"] == 0, result


if __name__ == "__main__":
    print(json.dumps(scenario()))
