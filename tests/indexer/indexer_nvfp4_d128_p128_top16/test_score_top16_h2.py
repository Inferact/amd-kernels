# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode top-16 at two index heads (TP2: 2 index-q heads per GPU over the replicated index-K head): every
two-head scorer (pinned) writes ``nvfp4_index_score_decode``'s bits below each row's block count, the ids equal
``reference.topk_reference`` on both selectors, graph replay at other lengths (CUDA-graph padding requests of length
0), and the one-head dispatch is the TP4 one."""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK, decode_kv_lens, score_width, topk_reference
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    host,
    nvfp4_index_score_decode,
    nvfp4_index_score_rows,
    nvfp4_index_score_top16,
    nvfp4_index_top16_from_scores,
)

from .inputs import index_cache, queries, require_gfx950

MAX_MODEL_LEN = 1 << 20
HEADS = 2
SCORERS = ["page", "first_page_xcd", "runs", "chunked", None]


def tp4_fast_scorer(reqs, ql, heads):
    """The one-head scorer choice of the TP4 dispatch."""
    if heads != 1 or ql not in (4, 8) or not 1 <= reqs <= 16:
        return None
    if reqs in (1, 3):
        return "page"
    if reqs == 2:
        return "first_page_xcd" if ql == 8 else "first_page"
    return "sector" if reqs == 4 and ql == 8 else "scan"


def test_one_head_dispatch_unchanged():
    for ql in (1, 2, 4, 8):
        for reqs in range(0, 300):
            assert host._fast_scorer(reqs, ql, 1) == tp4_fast_scorer(reqs, ql, 1), (reqs, ql)
            rows = reqs * ql
            assert host._plain_selector(rows, ql, 1) == (ql in (4, 8) and 0 < rows <= 128)


def test_two_head_dispatch():
    want = {1: "page", 2: "first_page_xcd", 12: "first_page_xcd", 13: "runs", 64: "runs", 256: "runs", 257: None}
    for ql in (4, 8):
        for reqs, kind in want.items():
            assert host._fast_scorer(reqs, ql, HEADS) == kind, (reqs, ql)
    assert host._fast_scorer(8, 7, HEADS) is None
    assert host._plain_selector(256, 8, HEADS) and not host._plain_selector(257, 8, HEADS)


def chunked(iq, cache, bt, seq, init, local, ql):
    rows = iq.shape[0]
    score = torch.full((HEADS, rows, score_width(MAX_MODEL_LEN)), float("nan"), device="cuda")
    nvfp4_index_score_decode(iq, cache, score, bt, seq, init, local, query_len=ql, max_seq_len=MAX_MODEL_LEN)
    return score


def want_ids(score, lens, ql):
    kv = torch.cat([decode_kv_lens(L, ql, "cuda") for L in lens])
    return topk_reference(score, (kv.long() + BLK - 1) // BLK, MAX_MODEL_LEN)


def check_bits(got, want, lens, ql):
    for r, length in enumerate(lens):
        nb = (length + BLK - 1) // BLK
        a = got[:, r * ql:(r + 1) * ql, :nb].view(torch.int32)
        b = want[:, r * ql:(r + 1) * ql, :nb].view(torch.int32)
        assert torch.equal(a, b), (r, length)


@pytest.fixture
def pin(monkeypatch):
    def set_pin(scorer=None, plain=None):
        monkeypatch.setattr(host, "PIN_SCORER", scorer)
        monkeypatch.setattr(host, "PIN_PLAIN", plain)

    return set_pin


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s or "auto")
@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("reqs", [1, 2, 3, 4, 5, 8, 12, 13, 16, 17, 33, 64, 128])
def test_scorer_bits_and_ids(scorer, ql, reqs, pin):
    require_gfx950()
    if scorer == "page" and reqs > 12 or scorer == "first_page_xcd" and reqs > 24:
        pytest.skip("a lane holds one strided page-table entry: at 1M, 12 / 24 requests at most")
    torch.manual_seed(reqs * 13 + ql)
    pages = 600
    cache = index_cache(pages)
    bt = torch.randint(0, pages, (reqs, MAX_MODEL_LEN // BLK), dtype=torch.int32, device="cuda")
    iq = queries(reqs * ql, HEADS)
    for lengths in ([131077, 1, 0, 33000], [7, 129, 212993, 2048], [0, 0, 0, 0], [1048576, 3, 128, 300001]):
        lens = (lengths * ((reqs + 3) // 4))[:reqs]
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        for init, local in ((0, 1), (2, 3)):
            old = chunked(iq, cache, bt, seq, init, local, ql)
            want = want_ids(old, lens, ql)
            for plain in (True, False):
                pin(scorer, plain)
                sc = torch.full_like(old, float("nan"))
                ids = torch.full_like(want, -7)
                nvfp4_index_score_top16(iq, cache, bt, seq, init, local, ids, query_len=ql,
                                        max_seq_len=MAX_MODEL_LEN, score=sc)
                assert torch.equal(ids, want), (lens, init, local, plain)
                check_bits(sc, old, lens, ql)
                again = torch.full_like(want, -7)
                nvfp4_index_score_rows(iq, cache, sc, bt, seq, init, local, query_len=ql, max_seq_len=MAX_MODEL_LEN)
                nvfp4_index_top16_from_scores(sc, seq, again, rows=reqs * ql, query_len=ql, max_seq_len=MAX_MODEL_LEN)
                assert torch.equal(again, want)


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("reqs", [1, 2, 3, 4, 8, 16, 24, 48])
def test_graph_replay_other_lengths(ql, reqs):
    require_gfx950()
    torch.manual_seed(5)
    pages = 2600
    cache = index_cache(pages)
    bt = torch.randint(0, pages, (reqs, MAX_MODEL_LEN // BLK), dtype=torch.int32, device="cuda")
    iq = queries(reqs * ql, HEADS)
    seq = torch.tensor(([40000, 9000, 1, 120000] * ((reqs + 3) // 4))[:reqs], dtype=torch.int32, device="cuda")
    score = torch.empty(HEADS, reqs * ql, score_width(MAX_MODEL_LEN), device="cuda")
    ids = torch.empty(HEADS, reqs * ql, 16, dtype=torch.int32, device="cuda")
    run = lambda: nvfp4_index_score_top16(  # noqa: E731
        iq, cache, bt, seq, 0, 1, ids, query_len=ql, max_seq_len=MAX_MODEL_LEN, score=score
    )
    run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for lens in ([100000, 50000, 200000, 1000], [0, 0, 0, 0], [7, 129, 1000, 1], [250000, 0, 3, 212993]):
        lens = (lens * ((reqs + len(lens) - 1) // len(lens)))[:reqs]
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        iq.view(torch.uint8).copy_(queries(reqs * ql, HEADS).view(torch.uint8))
        cache.random_(0, 256)
        score.fill_(float("nan"))
        ids.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(ids, want_ids(chunked(iq, cache, bt, seq, 0, 1, ql), lens, ql)), lens


@pytest.mark.parametrize("reqs", [2, 24])
def test_strided_query_takes_the_chunked_scorer(reqs):
    """A non-contiguous [rows, 2, 128] query (or score) is scored by ``nvfp4_index_score_decode``: same ids."""
    require_gfx950()
    torch.manual_seed(7)
    ql = 8
    cache = index_cache(300)
    bt = torch.randint(0, 300, (reqs, MAX_MODEL_LEN // BLK), dtype=torch.int32, device="cuda")
    wide = queries(reqs * ql, 3)
    iq = wide[:, :2]
    lens = ([33000, 7, 131077] * reqs)[:reqs]
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    ids = torch.full((HEADS, reqs * ql, 16), -7, dtype=torch.int32, device="cuda")
    nvfp4_index_score_top16(iq, cache, bt, seq, 0, 1, ids, query_len=ql, max_seq_len=MAX_MODEL_LEN)
    assert torch.equal(ids, want_ids(chunked(iq.contiguous(), cache, bt, seq, 0, 1, ql), lens, ql))


@pytest.mark.parametrize("reqs", [1, 4, 32])
def test_with_mfma_corunner(reqs):
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    torch.manual_seed(292)
    ql = 8
    iq = queries(reqs * ql, HEADS)
    cache = index_cache(64)
    bt = torch.randint(0, 64, (reqs, 8192), device="cuda", dtype=torch.int32)
    lens = ([131077, 7, 0, 300001] * reqs)[:reqs]
    seq = torch.tensor(lens, device="cuda", dtype=torch.int32)
    old = chunked(iq, cache, bt, seq, 2, 3, ql)
    want = want_ids(old, lens, ql)
    sc = torch.full_like(old, float("nan"))
    ids = torch.full_like(want, -7)

    def run():
        nvfp4_index_score_top16(iq, cache, bt, seq, 2, 3, ids, query_len=ql, max_seq_len=MAX_MODEL_LEN, score=sc)

    run()
    spin = mfma_spinner(256)
    high = torch.cuda.Stream(priority=-1)
    with torch.cuda.stream(high):
        spin(100000, high)
    for _ in range(20):
        run()
    torch.cuda.synchronize()
    assert torch.equal(ids, want)
    check_bits(sc, old, lens, ql)
