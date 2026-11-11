# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The XCD-local unit-to-block order of the one-head first-page scorer at 8 rows x 2 requests
(``score_firstpage_d128_p128_h1`` with ``xcd``): units below splits rounded down to 256 score block 256 (y // 32) +
32 (x % 8) + y % 32 (y = x // 8), so one 128-B score line comes from one XCD; the rest keep their block. Scores
bitwise ``nvfp4_index_score_decode`` below every row's block count, ids bit for bit the two-call chain and
``reference.topk_reference``, next to the plain order at other shapes; split counts that are a multiple of 256 and
not, rows past one pass of the units and past one selector window, padding rows, graph replay, which scorer runs."""

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    decode_kv_lens,
    score_width,
    topk_reference,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_decode,
    nvfp4_index_score_rows,
    nvfp4_index_score_top16,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import (
    index_topk_decode_indices,
)

from .inputs import block_table, index_cache, queries, require_gfx950

QL = 8
M1 = 1 << 20

CASES = [
    # one request: the plain page scorer next to it
    ([131072], 0, 1, M1),
    ([131073], 0, 1, M1),
    ([32768 + 5], 2, 3, M1),
    ([300000], 0, 1, M1),
    ([1], 0, 1, M1),
    ([65536], 0, 1, 100000),
    # two requests: 1536 units, rows past one pass and past one selector window
    ([204800, 215040], 0, 1, M1),
    ([300000, 65536 + 9], 0, 1, M1),
    ([1, 0], 0, 1, M1),
    ([131072 + 77, 127], 0, 1, M1),
    ([65536 + 9] * 2, 2, 3, M1),
    ([65536, 40000 + 3], 0, 1, 1 << 16),  # 512 units, still a multiple of 256
    ([100000, 7], 0, 1, 100000),  # 782 units: the plain order
    # three requests: 512 units, up to five passes
    ([196608, 1, 0], 0, 1, M1),
    ([215040, 204800, 70001], 0, 1, M1),
    ([300000, 3, 128 * 512 + 5], 0, 1, M1),
    ([65536 + 9] * 3, 2, 3, M1),
    ([65536, 1, 30000], 0, 1, 1 << 16),
    ([1000, 900], 0, 1, 1024),  # 8 units: none remapped
]


def setup(lens, seed, ql=QL):
    torch.manual_seed(seed)
    bt, pages = block_table(lens)
    cache = index_cache(pages)
    iq = queries(len(lens) * ql, 1)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return bt, cache, iq, seq


def block_counts(lens, ql=QL):
    kv = torch.cat([decode_kv_lens(L, ql, "cuda") for L in lens])
    return (kv.long() + BLK - 1) // BLK


def check(lens, init, local, hmax, bt, cache, iq, seq, ql=QL):
    rows, width = iq.shape[0], score_width(hmax)
    nblk = block_counts(lens, ql)
    got = torch.full((1, rows, width), float("nan"), device="cuda")
    want = torch.empty(1, rows, width, device="cuda")
    nvfp4_index_score_rows(iq, cache, got, bt, seq, init, local, query_len=ql, max_seq_len=hmax)
    nvfp4_index_score_decode(iq, cache, want, bt, seq, init, local, query_len=ql, max_seq_len=hmax)
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    sc = torch.empty(1, rows, width, device="cuda")
    nvfp4_index_score_top16(iq, cache, bt, seq, init, local, ids, query_len=ql, max_seq_len=hmax, score=sc)
    ref_ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    index_topk_decode_indices(want, ref_ids, seq, max_seq_len=hmax, query_len=ql)
    torch.cuda.synchronize()
    live = torch.arange(width, device="cuda")[None, :] < nblk.clamp(max=width)[:, None]
    g, w = got[0].view(torch.int32), want[0].view(torch.int32)
    assert torch.equal(torch.where(live, g, 0), torch.where(live, w, 0))
    assert torch.equal(ids, ref_ids)
    assert torch.equal(ids, topk_reference(want, nblk, hmax))
    return ids


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("lens,init,local,hmax", CASES)
def test_xcd_order_bitwise(lens, init, local, hmax, ql):
    require_gfx950()
    bt, cache, iq, seq = setup(lens, len(lens) * 31 + init + hmax % 97 + ql, ql)
    ids = check(lens, init, local, hmax, bt, cache, iq, seq, ql)
    again = torch.full_like(ids, -7)
    sc = torch.empty(1, iq.shape[0], score_width(hmax), device="cuda")
    nvfp4_index_score_top16(iq, cache, bt, seq, init, local, again, query_len=ql, max_seq_len=hmax, score=sc)
    torch.cuda.synchronize()
    assert torch.equal(again, ids)  # deterministic


@pytest.mark.parametrize("capture_lens,replay_lens", [
    ([131072], [131073]),
    ([196608], [32768 + 5]),
    ([204800, 215040], [1, 0]),
    ([65536, 3], [300000, 131072 + 77]),
    ([196608, 1, 0], [215040, 0, 70001]),
    ([7, 7, 7], [300000, 204800, 128 * 512 + 5]),
    ([131072, 131073, 1], [131073, 131072, 0]),
])
def test_xcd_order_graph_replay(capture_lens, replay_lens):
    require_gfx950()
    lens_max = [max(a, b) for a, b in zip(capture_lens, replay_lens)]
    bt, cache, iq, seq = setup(lens_max, 7 * len(lens_max))
    rows, width = iq.shape[0], score_width(M1)
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    sc = torch.empty(1, rows, width, device="cuda")
    seq.copy_(torch.tensor(capture_lens, dtype=torch.int32))

    def run():
        nvfp4_index_score_top16(iq, cache, bt, seq, 0, 1, ids, query_len=QL, max_seq_len=M1, score=sc)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    seq.copy_(torch.tensor(replay_lens, dtype=torch.int32))
    ids.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    got = ids.clone()
    want = check(replay_lens, 0, 1, M1, bt, cache, iq, seq)
    assert torch.equal(got, want)


@pytest.mark.parametrize("reqs,ql,hmax,xcd", [
    (2, 8, M1, True), (2, 8, 100000, True), (2, 8, 1024, True), (3, 8, M1, False),
    (1, 8, M1, False), (4, 8, M1, False), (2, 4, M1, False), (3, 4, M1, False),
])
def test_xcd_order_launched(reqs, ql, hmax, xcd):
    """The XCD order runs exactly at 8 rows x 2 requests, at every host length (one kernel, which the 1M warm-up
    covers)."""
    from torch.profiler import ProfilerActivity, profile
    require_gfx950()
    iq = queries(reqs * ql, 1)
    cache = torch.zeros(2, 128, 72, dtype=torch.uint8, device="cuda")
    bt = torch.zeros(reqs, 8192, dtype=torch.int32, device="cuda")
    seq = torch.full((reqs,), ql, dtype=torch.int32, device="cuda")
    sc = torch.empty(1, reqs * ql, score_width(hmax), device="cuda")
    nvfp4_index_score_rows(iq, cache, sc, bt, seq, 0, 1, query_len=ql, max_seq_len=hmax)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        nvfp4_index_score_rows(iq, cache, sc, bt, seq, 0, 1, query_len=ql, max_seq_len=hmax)
        torch.cuda.synchronize()
    names = [e.key for e in prof.key_averages() if "index_score" in e.key]
    assert len(names) == 1, names
    assert ("_xcd" in names[0]) == xcd, names
