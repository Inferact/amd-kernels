# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode index scorer (``score_decode.nvfp4_index_score_decode``) against ``reference.py``: every route the host
picks, graph replay, determinism. Served calls (fp8 keys): 1 index head x 4 rows, 1..256 requests (the chunked
shared-page kernel, plain or buffered by the score footprint).
"""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    SCORE_ATOL,
    SCORE_RTOL,
    cdiv,
    decode_scores_reference,
    score_width,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_decode
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host

from .inputs import (
    assert_scores,
    block_table,
    decode_lens,
    index_cache,
    nan_like,
    queries,
    require_gfx950,
)

INIT, LOCAL = 0, 1  # the model's sparse_init_block / sparse_local_block
MAX_MODEL_LEN = 1 << 20  # what decode graphs are captured with


def record_routes(monkeypatch) -> list[str]:
    """The decode kernels the host launches from now on: "chunked", "chunked_buf"."""
    routes: list[str] = []
    get_kernel = host._get_kernel

    def record_kernel(rows, heads, buffered):
        routes.append("chunked" + ("_buf" if buffered else ""))
        return get_kernel(rows, heads, buffered)

    monkeypatch.setattr(host, "_get_kernel", record_kernel)
    return routes


def make_case(lens, heads, ql, page_step=1, share=False, saturate=False):
    table, pages = block_table(lens, share)
    cache = index_cache(pages, page_step, saturate)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = queries(len(lens) * ql, heads)
    return q, cache, table, seq


def check(score, q, cache, table, lens, ql, init=INIT, local=LOCAL):
    """Every request's blocks against the reference; nothing past them written
    (``score`` NaN-filled before the call)."""
    want = decode_scores_reference(q, cache, table, lens, init, local, ql)
    stats = []
    for r, s in enumerate(lens):
        nb = cdiv(s, BLK)
        sl = slice(r * ql, (r + 1) * ql)
        assert score[:, sl, nb:].isnan().all(), (r, s)
        if nb:
            stats.append(
                assert_scores(
                    score[:, sl, :nb],
                    want[r],
                    SCORE_RTOL,
                    SCORE_ATOL,
                    (r, s),
                )
            )
    if stats:
        rel = max(x[0] for x in stats)
        l2 = max(x[1] for x in stats)
        cos = min(x[2] for x in stats)
        print(f"max rel {rel:.2e}, max rel L2 {l2:.2e}, min cos {cos:.9f}")
    return stats


def run(q, cache, table, seq, ql, max_seq_len, width, **kw):
    score = nan_like(q.shape[1], q.shape[0], width)
    nvfp4_index_score_decode(q, cache, score, table, seq, INIT, LOCAL, ql, max_seq_len, **kw)
    return score


# (heads, query rows, requests, length bound, page step, shared prefixes, route)
SERVED = [
    (1, 4, 1, 9000, 1, False, "chunked"),
    (1, 4, 5, 9000, 2, True, "chunked"),
    (1, 4, 40, 9000, 1, True, "chunked"),
    (1, 4, 70, 9000, 1, False, "chunked"),
    (1, 4, 130, 6000, 2, True, "chunked"),
]


@pytest.mark.parametrize("captured", [False, True], ids=["eager", "max_model_len"])
@pytest.mark.parametrize(
    "heads,ql,reqs,hi,page_step,share,route",
    SERVED,
    ids=[f"h{c[0]}q{c[1]}-n{c[2]}-ps{c[4]}{'-shared' if c[5] else ''}" for c in SERVED],
)
@torch.inference_mode()
def test_decode_served(heads, ql, reqs, hi, page_step, share, route, captured, monkeypatch):
    """The served call at the batch's own max_seq_len and at max_model_len: the route, and the scores against the
    reference."""
    require_gfx950()
    routes = record_routes(monkeypatch)
    torch.manual_seed(reqs * 10 + ql)
    lens = decode_lens(reqs, hi, reqs)
    q, cache, table, seq = make_case(lens, heads, ql, page_step, share)
    msl = MAX_MODEL_LEN if captured else max(lens)
    width = cdiv(max(lens), BLK) + 2
    score = run(q, cache, table, seq, ql, msl, width)
    torch.cuda.synchronize()
    assert routes[0].startswith(route), routes
    check(score, q, cache, table, lens, ql)


@torch.inference_mode()
def test_decode_saturating_keys():
    """Keys past 448 clamp on the fp8 route (``e4m3_rn_satfinite``)."""
    require_gfx950()
    torch.manual_seed(5)
    lens = decode_lens(9, 4000, 9)
    q, cache, table, seq = make_case(lens, 1, 8, saturate=True)
    score = run(q, cache, table, seq, 8, max(lens), cdiv(max(lens), BLK))
    torch.cuda.synchronize()
    check(score, q, cache, table, lens, 8)


# every route, non-served layouts included
ROUTE_SHAPES = [(1, 1), (1, 4), (1, 8), (2, 4)]


@pytest.mark.parametrize("buffered", [False, True], ids=["direct", "buffered"])
@pytest.mark.parametrize("num_reqs", [1, 40, 130])
@pytest.mark.parametrize("heads,ql", ROUTE_SHAPES, ids=[f"h{h}q{q}" for h, q in ROUTE_SHAPES])
@torch.inference_mode()
def test_decode_routes(heads, ql, num_reqs, buffered, monkeypatch):
    """The scorer on every route: chunks of up to 64 requests (padded columns for rows not a power of two) and the
    buffered variants (forced)."""
    require_gfx950()
    if buffered:
        monkeypatch.setattr(host, "BUFFER_MIN_BYTES", 0)
        monkeypatch.setattr(host, "UNITS_PER_SLOT", 1)
    torch.manual_seed(num_reqs)
    lens = decode_lens(num_reqs, 6000, num_reqs)
    q, cache, table, seq = make_case(lens, heads, ql, 1 + num_reqs % 2, True)
    width = cdiv(max(lens), BLK) + 2
    actual = run(q, cache, table, seq, ql, max(lens), width)
    torch.cuda.synchronize()
    check(actual, q, cache, table, lens, ql)


@pytest.mark.parametrize("heads,ql,reqs", [(1, 4, 40), (1, 4, 130)], ids=str)
@torch.inference_mode()
def test_decode_graph_replay(heads, ql, reqs):
    """The served call captured once at max_model_len, replayed on new queries, lengths and block tables written
    into the captured buffers (scores NaN-filled before each replay)."""
    require_gfx950()
    torch.manual_seed(reqs + heads)
    hi = 8000
    lens0 = decode_lens(reqs, hi, 1)
    table, pages = block_table([hi] * reqs)
    cache = index_cache(pages)
    seq = torch.tensor(lens0, dtype=torch.int32, device="cuda")
    q = queries(reqs * ql, heads)
    width = score_width(MAX_MODEL_LEN)
    score = nan_like(heads, reqs * ql, width)

    def call():
        nvfp4_index_score_decode(q, cache, score, table, seq, INIT, LOCAL, ql, MAX_MODEL_LEN)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for seed in (2, 3):
        lens = decode_lens(reqs, hi, seed)
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        q.copy_(queries(reqs * ql, heads))
        table.copy_(table[torch.randperm(reqs, device="cuda")])
        score.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(score, q, cache, table, lens, ql)


@pytest.mark.parametrize("heads,ql,reqs,msl", [(1, 4, 130, 1 << 17)], ids=str)
@torch.inference_mode()
def test_decode_deterministic(heads, ql, reqs, msl):
    """Three launches give the same scores bit for bit."""
    require_gfx950()
    torch.manual_seed(31)
    lens = decode_lens(reqs, 100000, 31)
    q, cache, table, seq = make_case(lens, heads, ql, 1, True)
    width = score_width(msl)
    outs = [
        run(q, cache, table, seq, ql, msl, width).view(torch.int32)
        for _ in range(3)
    ]
    torch.cuda.synchronize()
    assert all(torch.equal(outs[0], o) for o in outs[1:])
