# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-request scorer's scalar first entry and device-bounded future entries."""
import pytest
import torch

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_decode,
    nvfp4_index_score_top16,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import index_topk_decode_indices
from .inputs import index_cache, queries, require_gfx950


@pytest.mark.parametrize("ql", [4, 8])
@pytest.mark.parametrize("init,local", [(0, 1), (2, 3)])
def test_first_and_future_entries_changed_length_graph(ql, init, local):
    """Cross the first/future entry at 1536 blocks, then revisit it after padding."""
    require_gfx950()
    torch.manual_seed(20261107 + ql)
    hmax = 1 << 20
    width = score_width(hmax)
    cache = index_cache(64, page_step=2)
    q = queries(4 * ql, 1)[::2]
    table = torch.randint(0, 64, (2, width + 7), device="cuda", dtype=torch.int32)[:, :width]
    seq = torch.full((2,), 32768, device="cuda", dtype=torch.int32)
    work = torch.full((1, 2 * ql, width + 5), float("nan"), device="cuda")
    score = work[:, :, :width]
    old = torch.empty_like(work)[:, :, :width]
    ids = torch.full((1, 2 * ql, 16), -7, device="cuda", dtype=torch.int32)
    want = torch.empty_like(ids)

    def run():
        nvfp4_index_score_top16(q, cache, table, seq, init, local, ids,
            query_len=ql, max_seq_len=hmax, score=score)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    lengths = [
        [196608 - ql, 196608 - 1],
        [196608, 196609],
        [hmax, 524289],
        [0, 0],
        [1, ql - 1],
        [196609, 0],
    ]
    for lens in lengths:
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        # Replays may not retain the first entry, later entries, Q, or LUT values.
        table.random_(0, 64)
        q.view(torch.uint8).copy_(queries(2 * ql, 1).view(torch.uint8))
        cache.random_(0, 256)
        work.fill_(float("nan"))
        old.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        nvfp4_index_score_decode(q, cache, old, table, seq, init, local,
            query_len=ql, max_seq_len=hmax)
        index_topk_decode_indices(old, want, seq, max_seq_len=hmax, query_len=ql)
        assert torch.equal(ids, want), lens
        for req, length in enumerate(lens):
            nb = (length + 127) // 128
            sl = (slice(None), slice(req * ql, (req + 1) * ql), slice(0, nb))
            assert torch.equal(score[sl].view(torch.int32), old[sl].view(torch.int32)), lens
            assert torch.isnan(score[:, req * ql:(req + 1) * ql, nb:]).all(), lens
        assert torch.isnan(work[:, :, width:]).all()
        previous = ids.clone()
        run()
        assert torch.equal(ids, previous), lens
