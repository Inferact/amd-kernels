# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Against AITER's MSA kernels in the image (``aiter.ops.msa_attention``): the top-k (decode: indices; prefill:
indices, page table, context bounds) equals ``pa_sparse_block_topk`` bit for bit; the fp8-route scorers share the FP8
products of ``pa_sparse_block_score_decode`` / ``_prefill`` over the same keys as FP8 pages
(``e4m3_rn_satfinite(e2m1 * scale)``), so only the fp32 summation order differs."""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK, cdiv, fp8_index_pages
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_decode
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import nvfp4_index_score_prefill
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import index_topk_decode_indices

from .inputs import block_table, index_cache, nan_like, queries, require_gfx950
from .test_topk import (
    KINDS,
    PREFILL_IDS,
    PREFILL_SHAPES,
    SHAPE_IDS,
    SHAPES,
    decode_case,
    outputs,
    prefill_case,
    run_prefill,
)

msa = pytest.importorskip("aiter.ops.msa_attention")


def aiter_decode(c):
    heads, rows, _ = c["score"].shape
    views, compared, _ = outputs(heads, rows, c["nkv"], False)
    msa.pa_sparse_block_topk(
        c["score"], views[0], c["bt"], c["seq"], views[1], views[2],
        max_seq_len=c["msl"], block_size=BLK, query_len=c["ql"],
        num_kv_heads=c["nkv"], pages_per_block=8,
    )  # fmt: skip
    return compared


def aiter_prefill(c):
    heads, rows, _ = c["score"].shape
    views, compared, _ = outputs(heads, rows, c["nkv"], False)
    msa.pa_sparse_block_topk(
        c["score"], views[0], c["bt"], c["seq"], views[1], views[2],
        max_seq_len=c["msl"], block_size=BLK, num_valid_pages=c["nvp"],
        row_req_id=c["rid"], kv_lens=c["kvl"], num_kv_heads=c["nkv"], pages_per_block=8,
    )  # fmt: skip
    return compared


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@torch.inference_mode()
def test_topk_decode_equals_aiter(kind, shape):
    """The decode indices equal AITER's ``topk_idx``."""
    require_gfx950()
    c = decode_case(kind, shape)
    want = aiter_decode(c)[0]
    got = torch.full_like(want, -7)
    index_topk_decode_indices(c["score"], got, c["seq"], max_seq_len=c["msl"], query_len=c["ql"])
    torch.cuda.synchronize()
    torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", PREFILL_SHAPES, ids=PREFILL_IDS)
@torch.inference_mode()
def test_topk_prefill_equals_aiter(kind, shape):
    require_gfx950()
    c = prefill_case(kind, shape)
    want = aiter_prefill(c)
    _, got, _ = run_prefill(c)
    torch.cuda.synchronize()
    for name, g, w in zip(("topk_idx", "sparse_bt", "sparse_ctx"), got, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)


# (heads, rows)
DECODE = [(1, 4), (1, 1)]


@pytest.mark.parametrize("heads,ql", DECODE, ids=[f"h{h}q{q}" for h, q in DECODE])
@torch.inference_mode()
def test_decode_scores_share_aiter_products(heads, ql):
    """On every block a row can see; sentinels identical."""
    require_gfx950()
    torch.manual_seed(0)
    lens = [5, 300, 2100, 9000, 129, 40000]
    table, pages = block_table(lens)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    cache = index_cache(pages)
    rows = len(lens) * ql
    q = queries(rows, heads)
    msl = max(lens)
    width = cdiv(msl, BLK)
    actual = nan_like(heads, rows, width)
    nvfp4_index_score_decode(q, cache, actual, table, seq, 1, 2, ql, msl)
    expected = nan_like(heads, rows, width)
    pages8 = fp8_index_pages(cache)
    for h0 in range(0, heads, 2):  # AITER builds one / two-head scorers
        msa.pa_sparse_block_score_decode(
            q[:, h0 : h0 + 2].contiguous(), pages8, expected[h0 : h0 + 2], table, seq,
            init_blocks=1, local_blocks=2, query_len=ql, max_seq_len=msl,
        )  # fmt: skip
    torch.cuda.synchronize()
    for r, s in enumerate(lens):
        for t in range(ql):
            row = r * ql + t
            seen = cdiv(max(s - ql + t + 1, 0), BLK)
            got, want = actual[:, row, :seen], expected[:, row, :seen]
            finite = want < 1e28
            assert torch.equal(got[~finite], want[~finite])
            torch.testing.assert_close(got[finite], want[finite], rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize(
    "q_lens,seq_lens",
    [([129, 300, 5], [129, 2500, 2049]), ([1000, 700], [4300, 700]), ([3000, 700], [20000, 30000])],
    ids=["small", "unaligned", "tile512"],
)
@torch.inference_mode()
def test_prefill_scores_share_aiter_products(q_lens, seq_lens, heads):
    """On every block a row can see (the top-k reads nothing else); sentinels
    identical."""
    require_gfx950()
    torch.manual_seed(0)
    table, pages = block_table(seq_lens)
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device="cuda").cumsum(0)
    cache = index_cache(pages)
    q = queries(sum(q_lens), heads)
    msl = max(seq_lens)
    width = cdiv(msl, BLK)
    actual = nan_like(heads, sum(q_lens), width)
    nvfp4_index_score_prefill(q, cache, actual, table, cu, seq, 1, 2, max(q_lens), msl)
    expected = nan_like(heads, sum(q_lens), width)
    pages8 = fp8_index_pages(cache)
    for h in range(heads):
        msa.pa_sparse_block_score_prefill(
            q[:, h : h + 1].contiguous(), pages8, expected[h : h + 1], table, cu, seq,
            init_blocks=1, local_blocks=2, max_query_len=max(q_lens), max_seq_len=msl,
        )  # fmt: skip
    torch.cuda.synchronize()
    q0 = 0
    for ql, s in zip(q_lens, seq_lens):
        nb = cdiv(s, BLK)
        rows = slice(q0, q0 + ql)
        pos = s - ql + torch.arange(ql, device="cuda")
        seen = (torch.arange(nb, device="cuda")[None, :] <= (pos // BLK)[:, None])[None]
        got = actual[:, rows, :nb][seen.expand(heads, -1, -1)]
        want = expected[:, rows, :nb][seen.expand(heads, -1, -1)]
        finite = want < 1e28
        assert torch.equal(got[~finite], want[~finite])
        torch.testing.assert_close(got[finite], want[finite], rtol=1e-5, atol=1e-4)
        q0 += ql
