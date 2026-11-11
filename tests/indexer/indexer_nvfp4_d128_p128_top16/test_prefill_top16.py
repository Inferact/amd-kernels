# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prefill top-16 entry (``prefill_top16.index_prefill_top16``): the ids against ``reference.py`` (fp64 scores of
the fp8-route keys, then ``topk_reference``; a set may differ only by blocks within twice the score tolerance of the
16th score), bit for bit the scorer + top-k called one by one, a caller's workspace, graph replay, the support check.
"""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import prefill_top16
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    SCORE_ATOL,
    SCORE_RTOL,
    cdiv,
    prefill_scores_reference,
    request_keys,
    score_width,
    topk_reference,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import nvfp4_index_score_prefill
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import index_topk_prefill_indices

from .inputs import block_table, index_cache, queries, require_gfx950

INIT, LOCAL = 0, 1  # the model's sparse_init_block / sparse_local_block
SLICE = 512  # reference rows per slice


def make_case(q_lens, seq_lens, heads):
    table, pages = block_table(seq_lens)  # pages out of order
    cache = index_cache(pages)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device="cuda").cumsum(0)
    pos = torch.cat([s - ql + torch.arange(ql, device="cuda") for ql, s in zip(q_lens, seq_lens)])
    nvp = (pos // BLK + 1).to(torch.int32)  # each row's causal blocks
    return queries(sum(q_lens), heads), cache, table, cu, seq, pos, nvp


def entry(q, cache, table, cu, seq, q_lens, seq_lens, nvp, init=INIT, local=LOCAL, score=None):
    out = torch.full((q.shape[1], q.shape[0], 16), -7, dtype=torch.int32, device="cuda")
    got = prefill_top16.index_prefill_top16(
        q, cache, table, cu, seq, init, local, out,
        max_query_len=max(q_lens), max_seq_len=max(seq_lens), num_valid_pages=nvp, score=score,
    )  # fmt: skip
    assert got is out
    return out


def check_reference(out, q, cache, table, q_lens, seq_lens, pos, nvp, init=INIT, local=LOCAL):
    """Every row's ids against the reference top-16 of the fp64 scores; returns the rows with a near-tie swap."""
    heads = q.shape[1]
    msl = max(seq_lens)
    width = score_width(msl)
    ties = 0
    q0 = 0
    for req, (ql, s) in enumerate(zip(q_lens, seq_lens)):
        nb = cdiv(s, BLK)
        keys = request_keys(cache, table, req, nb, False).double()
        for r0 in range(0, ql, SLICE):
            r1 = min(ql, r0 + SLICE)
            rows = slice(q0 + r0, q0 + r1)
            want, _ = prefill_scores_reference(q[rows], keys, pos[rows], init, local)  # [H, R, nb]
            full = torch.full((heads, r1 - r0, width), -torch.inf, dtype=torch.float64, device="cuda")
            full[..., :nb] = want
            ref = topk_reference(full.float(), nvp[rows].long(), msl)  # [H, R, 16]
            got = out[:, rows]
            same = (got.sort(-1).values == ref.sort(-1).values).all(-1)
            # the reference's 16th score of each row (live blocks only)
            thr = torch.gather(full, 2, ref.clamp(min=0).long()).masked_fill(ref < 0, torch.inf).min(-1).values
            for h, i in (~same).nonzero().tolist():
                a, b = set(got[h, i].tolist()), set(ref[h, i].tolist())
                assert (-1 in a) == (-1 in b), (req, r0 + i, a, b)
                swapped = torch.tensor(sorted(x for x in a ^ b if x >= 0), device="cuda")
                t = float(thr[h, i])
                tol = 2 * (SCORE_ATOL + SCORE_RTOL * max(abs(t), 1.0))  # both scores within the tolerance
                assert bool(((full[h, i, swapped] - t).abs() <= tol).all()), (req, r0 + i, a ^ b, t)
                ties += 1
        q0 += ql
    return ties


SERVED = [
    # (q_lens, seq_lens): the scorer tile the host picks at one head
    ([4096], [4096]),  # a fresh prompt, small work: 128-row tiles
    ([4096], [36864]),  # 512-row tiles, the first top-k window tile
    ([2000, 513, 3000], [9000, 513, 20000]),  # ragged, unaligned prefixes
    ([8192], [139264]),  # a DBO micro-batch on a long prefix (the 20-strip window)
    ([1024], [300000]),  # past one 26-strip window
]


@pytest.mark.parametrize("q_lens,seq_lens", SERVED, ids=[f"{a}-{b}".replace(" ", "") for a, b in SERVED])
@torch.inference_mode()
def test_reference(q_lens, seq_lens):
    require_gfx950()
    heads = 1
    torch.manual_seed(sum(q_lens) + heads)
    q, cache, table, cu, seq, pos, nvp = make_case(q_lens, seq_lens, heads)
    out = entry(q, cache, table, cu, seq, q_lens, seq_lens, nvp)
    torch.cuda.synchronize()
    ties = check_reference(out, q, cache, table, q_lens, seq_lens, pos, nvp)
    print(f"near-tie rows {ties} of {heads * sum(q_lens)}")


@pytest.mark.parametrize("init_blocks,local_blocks", [(0, 1), (2, 2)])
@pytest.mark.parametrize("q_lens,seq_lens", [([129, 300, 5], [129, 2500, 2049]), ([3000, 700], [20000, 30000])],
                         ids=["small", "tile512"])  # fmt: skip
@torch.inference_mode()
def test_two_calls_bitwise(q_lens, seq_lens, init_blocks, local_blocks):
    """The ordered ids of the scorer, then the top-k, one by one; with and without a caller's workspace."""
    require_gfx950()
    torch.manual_seed(3)
    q, cache, table, cu, seq, pos, nvp = make_case(q_lens, seq_lens, 1)
    msl = max(seq_lens)
    score = torch.full((1, q.shape[0], score_width(msl)), float("nan"), device="cuda")
    nvfp4_index_score_prefill(q, cache, score, table, cu, seq, init_blocks, local_blocks, max(q_lens), msl)
    want = torch.full((1, q.shape[0], 16), -7, dtype=torch.int32, device="cuda")
    index_topk_prefill_indices(score, want, max_seq_len=msl, block_size=BLK, num_valid_pages=nvp)
    got = entry(q, cache, table, cu, seq, q_lens, seq_lens, nvp, init_blocks, local_blocks)
    work = torch.empty(1, q.shape[0], score_width(msl) + 64, device="cuda")[..., : score_width(msl)]
    got_ws = entry(q, cache, table, cu, seq, q_lens, seq_lens, nvp, init_blocks, local_blocks, score=work)
    torch.cuda.synchronize()
    assert torch.equal(got, want) and torch.equal(got_ws, want)
    check_reference(got, q, cache, table, q_lens, seq_lens, pos, nvp, init_blocks, local_blocks)


@pytest.mark.parametrize("q_lens,seq_lens", [([129, 300], [129, 2500]), ([3000, 700], [20000, 30000])],
                         ids=["small", "tile512"])  # fmt: skip
@torch.inference_mode()
def test_graph_replay(q_lens, seq_lens):
    """Captured once (its score workspace allocated in the capture), replayed on new queries and block tables."""
    require_gfx950()
    torch.manual_seed(7)
    q, cache, table, cu, seq, pos, nvp = make_case(q_lens, seq_lens, 1)
    out = torch.empty(1, q.shape[0], 16, dtype=torch.int32, device="cuda")

    def call():
        prefill_top16.index_prefill_top16(
            q, cache, table, cu, seq, INIT, LOCAL, out,
            max_query_len=max(q_lens), max_seq_len=max(seq_lens), num_valid_pages=nvp,
        )  # fmt: skip

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        q.copy_(queries(q.shape[0], 1))
        perm = torch.randperm(cache.shape[0], device="cuda", dtype=torch.int32)
        table.copy_(perm[table.long()])
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        want = entry(q, cache, table, cu, seq, q_lens, seq_lens, nvp)
        assert torch.equal(out, want)


def test_blocker():
    require_gfx950()
    served = dict(num_index_heads=1, index_head_dim=128, block_size=128, topk_blocks=16)
    assert prefill_top16.index_prefill_top16_blocker(**served) is None
    for k, v in (("num_index_heads", 8), ("index_head_dim", 64), ("block_size", 64), ("topk_blocks", 32)):
        why = prefill_top16.index_prefill_top16_blocker(**{**served, k: v})
        assert why and str(v) in why, (k, why)
