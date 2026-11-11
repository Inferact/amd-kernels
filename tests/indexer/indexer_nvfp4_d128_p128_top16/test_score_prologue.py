# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The score-only entry (``score_decode.nvfp4_index_score_rows``) and a consumer kernel that runs the top-16 in its
prologue (``topk.top16_indices_p128_plain_wave4.row_top16_prologue_ids``): scores bitwise ``nvfp4_index_score_decode``
below every row's block count; ids bit for bit ``index_topk_decode_indices`` and ``reference.topk_reference`` with
one or several workgroups per row; graph replay at other lengths; determinism. Served shapes: one index head x 8 or
4 rows, the host at max_model_len."""

import functools

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
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import (
    index_topk_decode_indices,
)

from .inputs import block_table, index_cache, queries, require_gfx950

MAX_MODEL_LEN = 1 << 20  # what decode graphs are captured with

CASES = [
    ([32768], 0, 1),
    ([131072 + 77, 5000], 0, 1),
    ([196608, 1, 0], 0, 1),
    ([1, 2, 3, 300], 0, 1),
    ([65536 + 9] * 4, 2, 3),
    ([300000], 0, 1),  # past one 1792-block window
    ([4096 + 3] * 17, 0, 1),  # past the one-head scorers: the general scorer
]


@functools.cache
def _consumer(splits: int):
    """A consumer stand-in: grid (rows, splits) x 256 threads, each workgroup runs the prologue on its row and keeps
    the ids it stores (equal bytes from every workgroup)."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk.prefill_kernel import (
        SEL_DWORDS,
    )
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk.top16_indices_p128_plain_wave4 import (
        CAND_DWORDS,
        row_top16_prologue_ids,
    )

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, CAND_DWORDS, 16]
        sel: fx.Array[fx.Int32, SEL_DWORDS, 16]
        tmp: fx.Array[fx.Int32, 260, 16]

    @flyc.kernel(name=f"test_row_top16_prologue_s{splits}", known_block_size=[256, 1, 1])
    def kernel(arg_score: fx.Int64, s_row: fx.Int64, width: fx.Int32, arg_seq: fx.Int64,
               ql: fx.Int32, arg_topk: fx.Int64, t_row: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, tx = fx.block_idx.x, fx.thread_idx.x
        row_top16_prologue_ids(
            tx, arg_score + n.to(fx.Int64) * s_row * 4, width, arg_seq, n, ql,
            smem.cand.ptr, smem.sel.ptr, smem.tmp.ptr, arg_topk + n.to(fx.Int64) * t_row * 4,
        )

    @flyc.jit
    def launch(arg_score: fx.Int64, s_row: fx.Int64, width: fx.Int32, arg_seq: fx.Int64,
               ql: fx.Int32, arg_topk: fx.Int64, t_row: fx.Int64, rows: fx.Int32,
               stream: fx.Stream):
        kernel(arg_score, s_row, width, arg_seq, ql, arg_topk, t_row).launch(
            grid=(fx.Int64(rows), splits, 1), block=(256, 1, 1), stream=stream)

    return launch


def consumer(score, seq, ids, ql, hmax, splits):
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.host_stream import current_stream
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode.host import _run_compiled

    _run_compiled(
        _consumer(splits), score.data_ptr(), int(score.stride(1)), score_width(hmax),
        seq.data_ptr(), int(ql), ids.data_ptr(), int(ids.stride(1)), int(ids.shape[1]),
        current_stream(),
    )


def setup(lens, ql, seed):
    torch.manual_seed(seed)
    bt, pages = block_table(lens)
    cache = index_cache(pages)
    iq = queries(len(lens) * ql, 1)
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    kv = torch.cat([decode_kv_lens(L, ql, "cuda") for L in lens])
    nblk = (kv.long() + BLK - 1) // BLK
    return bt, cache, iq, seq, nblk


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("lens,init,local", CASES)
def test_scores_match_scorer(ql, lens, init, local):
    require_gfx950()
    bt, cache, iq, seq, nblk = setup(lens, ql, len(lens) * 17 + ql)
    rows, width = iq.shape[0], score_width(MAX_MODEL_LEN)
    got = torch.full((1, rows, width), float("nan"), device="cuda")
    want = torch.empty(1, rows, width, device="cuda")
    nvfp4_index_score_rows(iq, cache, got, bt, seq, init, local, query_len=ql,
                           max_seq_len=MAX_MODEL_LEN)
    nvfp4_index_score_decode(iq, cache, want, bt, seq, init, local, query_len=ql,
                             max_seq_len=MAX_MODEL_LEN)
    torch.cuda.synchronize()
    live = torch.arange(width, device="cuda")[None, :] < nblk.clamp(max=width)[:, None]
    g, w = got[0].view(torch.int32), want[0].view(torch.int32)
    assert torch.equal(torch.where(live, g, 0), torch.where(live, w, 0))


@pytest.mark.parametrize("ql", [8, 4])
@pytest.mark.parametrize("lens,init,local", CASES)
@pytest.mark.parametrize("splits", [1, 4])
def test_prologue_ids(ql, lens, init, local, splits):
    require_gfx950()
    bt, cache, iq, seq, nblk = setup(lens, ql, len(lens) * 29 + ql + init)
    rows, width = iq.shape[0], score_width(MAX_MODEL_LEN)
    score = torch.empty(1, rows, width, device="cuda")
    nvfp4_index_score_rows(iq, cache, score, bt, seq, init, local, query_len=ql,
                           max_seq_len=MAX_MODEL_LEN)
    want = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    index_topk_decode_indices(score, want, seq, max_seq_len=MAX_MODEL_LEN, query_len=ql)
    got = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    consumer(score, seq, got, ql, MAX_MODEL_LEN, splits)
    again = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    consumer(score, seq, again, ql, MAX_MODEL_LEN, splits)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    assert torch.equal(got, again)
    assert torch.equal(got, topk_reference(score, nblk, MAX_MODEL_LEN))


@pytest.mark.parametrize("ql", [8, 4])
def test_graph_replay_other_lengths(ql):
    """Score + prologue consumer captured at one set of lengths, replayed at others (padding requests too): ids
    equal the eager two calls at the replayed lengths."""
    require_gfx950()
    cap, other = [65536, 1000, 300], [196608 + 5, 0, 131072]
    bt, cache, iq, seq, _ = setup([max(a, b) for a, b in zip(cap, other)], ql, 5 + ql)
    rows, width = iq.shape[0], score_width(MAX_MODEL_LEN)
    score = torch.empty(1, rows, width, device="cuda")
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    seq.copy_(torch.tensor(cap, dtype=torch.int32))

    def step():
        nvfp4_index_score_rows(iq, cache, score, bt, seq, 0, 1, query_len=ql,
                               max_seq_len=MAX_MODEL_LEN)
        consumer(score, seq, ids, ql, MAX_MODEL_LEN, 4)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        step()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
    for lens in (other, cap, other):
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        ids.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        ws = torch.empty(1, rows, width, device="cuda")
        want = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
        nvfp4_index_score_decode(iq, cache, ws, bt, seq, 0, 1, query_len=ql,
                                 max_seq_len=MAX_MODEL_LEN)
        index_topk_decode_indices(ws, want, seq, max_seq_len=MAX_MODEL_LEN, query_len=ql)
        torch.cuda.synchronize()
        assert torch.equal(ids, want), lens
