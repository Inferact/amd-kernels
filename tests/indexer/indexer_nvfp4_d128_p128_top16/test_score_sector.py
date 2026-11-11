# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-block score sectors at four requests x eight rows: the complete-64-unit permutation, the identity tail,
every page pass, strided inputs and workspaces, exceptional score bits, changed-length graphs."""
import pytest
import torch

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import reference as ref
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_decode,
    nvfp4_index_score_rows,
    nvfp4_index_score_top16,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import index_topk_decode_indices
from .inputs import index_cache, queries, require_gfx950


def inputs(hmax):
    torch.manual_seed(hmax + 31)
    nb = (hmax + 127) // 128
    cache = index_cache(32, page_step=2)
    q = queries(64, 1)[::2]
    table = torch.randint(0, 32, (4, nb + 7), dtype=torch.int32, device='cuda')[:, :nb]
    seq = torch.zeros(4, dtype=torch.int32, device='cuda')
    width = ref.score_width(hmax)
    base = torch.full((1, 34, width + 17), float('nan'), device='cuda')
    score = base[:, :32, :width]
    ids = torch.full((1, 32, 16), -7, dtype=torch.int32, device='cuda')
    return q, cache, table, seq, base, score, ids


def run(q, cache, table, seq, score, ids, hmax, init, local):
    nvfp4_index_score_top16(q, cache, table, seq, init, local, ids,
                           query_len=8, max_seq_len=hmax, score=score)


def check(q, cache, table, seq, base, score, ids, hmax, init, local):
    old = torch.empty_like(score)
    nvfp4_index_score_decode(q, cache, old, table, seq, init, local,
                            query_len=8, max_seq_len=hmax)
    want = torch.full_like(ids, -7)
    index_topk_decode_indices(old, want, seq, max_seq_len=hmax, query_len=8)
    counts = torch.cat([ref.decode_kv_lens(n, 8, 'cuda') for n in seq.cpu().tolist()])
    counts = (counts.long() + 127) // 128
    live = torch.arange(score.shape[-1], device='cuda')[None, :] < counts[:, None]
    assert torch.equal(torch.where(live, score[0].view(torch.int32), 0),
                       torch.where(live, old[0].view(torch.int32), 0))
    assert torch.equal(ids, want)
    assert torch.equal(ids, ref.topk_reference(old, counts, hmax))
    assert torch.isnan(base[:, 32:]).all()
    assert torch.isnan(base[:, :32, score.shape[-1]:]).all()
    # the score-only entry runs the same scorer
    rows = torch.full_like(score, float('nan'))
    nvfp4_index_score_rows(q, cache, rows, table, seq, init, local,
                          query_len=8, max_seq_len=hmax)
    assert torch.equal(torch.where(live, rows[0].view(torch.int32), 0),
                       torch.where(live, old[0].view(torch.int32), 0))


@pytest.mark.parametrize('hmax', [1024, 8192, 8193, 128 * 129, 100000, 1 << 20])
@pytest.mark.parametrize('init,local', [(0, 1), (2, 3)])
def test_score_sector_boundaries(hmax, init, local):
    require_gfx950()
    q, cache, table, seq, base, score, ids = inputs(hmax)
    seq.copy_(torch.tensor([hmax, hmax - 7, 1, 0], dtype=torch.int32))
    run(q, cache, table, seq, score, ids, hmax, init, local)
    check(q, cache, table, seq, base, score, ids, hmax, init, local)
    saved = ids.clone()
    run(q, cache, table, seq, score, ids, hmax, init, local)
    assert torch.equal(ids, saved)


@pytest.mark.parametrize('kind', ['equal', 'zero', 'negative_zero', 'tiny', 'tiny_negative',
                                 'nan_q', 'nan_q_negative', 'nan_scale', 'nan_scale_negative'])
def test_score_sector_special_bits(kind):
    require_gfx950()
    hmax = 1 << 20
    q, cache, table, seq, base, score, ids = inputs(hmax)
    if kind == 'equal':
        cache[:] = cache[:1].clone()
    elif kind in ('zero', 'negative_zero', 'tiny', 'tiny_negative'):
        q.view(torch.uint8).fill_({'zero': 0, 'negative_zero': 0x80,
                                 'tiny': 1, 'tiny_negative': 0x81}[kind])
    elif kind.startswith('nan_q'):
        q.view(torch.uint8)[0, 0, 5] = 0xff if kind.endswith('negative') else 0x7f
    else:
        cache.as_strided((32, 9216), (cache.stride(0), 1))[:, 8192 + 5] = (
            0xff if kind.endswith('negative') else 0x7f)
    seq.copy_(torch.tensor([131077, 196608, 7, 0], dtype=torch.int32))
    run(q, cache, table, seq, score, ids, hmax, 2, 3)
    check(q, cache, table, seq, base, score, ids, hmax, 2, 3)


def test_score_sector_graph_inputs():
    require_gfx950()
    hmax = 1 << 20
    q, cache, table, seq, base, score, ids = inputs(hmax)
    seq.fill_(131072)
    run(q, cache, table, seq, score, ids, hmax, 2, 3)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(q, cache, table, seq, score, ids, hmax, 2, 3)
    for lens in ([0, 0, 0, 0], [1, 129, 8193, 7], [300001, 196608, 1048576, 0]):
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        q.view(torch.uint8).copy_(queries(32, 1).view(torch.uint8))
        cache.random_(0, 256)
        score.fill_(float('nan')); ids.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        check(q, cache, table, seq, base, score, ids, hmax, 2, 3)
