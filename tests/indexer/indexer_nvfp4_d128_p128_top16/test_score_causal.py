# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compact one-head scoring: causal task mapping, padding, and graph replay."""
import pytest
import torch

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import host
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK, cdiv
from .inputs import nan_like, queries, require_gfx950
from .test_score_prefill import make_case, check


@pytest.mark.parametrize('ql,prefix,maxq,maxseq', [
    (16384, 0, 16384, 16384), (16385, 1, 32768, 65536),
    (16384, 511, 32768, 65536), (16384, 512, 32768, 65536),
    (16384, 513, 32768, 65536), (16385, 65537, 32768, 131072),
])
@pytest.mark.parametrize('staged', [True, False])
@torch.inference_mode()
def test_causal_scores(ql, prefix, maxq, maxseq, staged, monkeypatch):
    require_gfx950()
    monkeypatch.setattr(host, 'PIN_SHARED', False)
    monkeypatch.setattr(host, 'PIN_TARGET', 32768)
    monkeypatch.setattr(host, 'STAGED_STORES', staged)
    torch.manual_seed(ql + prefix)
    q, cache, table, cu, seq = make_case([ql], [ql + prefix], 1, 2)
    width = cdiv(maxseq, BLK) + 3
    original, compact, repeat = [nan_like(1, ql, width) for _ in range(3)]
    def call(out, compact):
        host.nvfp4_index_score_prefill(q, cache, out, table, cu, seq, 2, 3,
            maxq, maxseq, causal_grid=compact)
    call(original, False)
    call(compact, True)
    call(repeat, True)
    torch.cuda.synchronize()
    assert torch.equal(original.nan_to_num(7.0).view(torch.int32), compact.nan_to_num(7.0).view(torch.int32))
    assert torch.equal(compact.nan_to_num(7.0).view(torch.int32), repeat.nan_to_num(7.0).view(torch.int32))
    check(compact, q, cache, table, [ql], [ql + prefix], init=2, local=3)


@torch.inference_mode()
def test_causal_graph_lengths(monkeypatch):
    require_gfx950()
    monkeypatch.setattr(host, 'PIN_SHARED', False)
    monkeypatch.setattr(host, 'PIN_TARGET', 32768)
    torch.manual_seed(71)
    rows, maxseq = 16384, 32768
    q, cache, table, cu, seq = make_case([rows], [maxseq], 1)
    out = nan_like(1, rows, cdiv(maxseq, BLK) + 3)
    def call():
        host.nvfp4_index_score_prefill(q, cache, out, table, cu, seq, 0, 1,
            rows, maxseq, causal_grid=True)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for ql, sl in [(8191, 8191), (513, 16383), (8192, 4097), (1, 513), (0, 0)]:
        cu[1] = ql
        seq[0] = sl
        q.copy_(queries(rows, 1))
        out.fill_(float('nan'))
        graph.replay()
        torch.cuda.synchronize()
        if ql:
            check(out[:, :ql], q[:ql], cache, table, [ql], [sl])
        assert torch.isnan(out[:, ql:]).all()


@torch.inference_mode()
def test_causal_next_to_mfma(monkeypatch):
    """The compact scorer also keeps its bits while MFMA work shares its CUs."""
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    monkeypatch.setattr(host, 'PIN_SHARED', False)
    monkeypatch.setattr(host, 'PIN_TARGET', 32768)
    torch.manual_seed(97)
    rows, seq_len = 16384, 16497
    q, cache, table, cu, seq = make_case([rows], [seq_len], 1)
    width = cdiv(seq_len, BLK) + 3
    want = nan_like(1, rows, width)
    outs = [nan_like(1, rows, width) for _ in range(8)]

    def call(out, compact):
        host.nvfp4_index_score_prefill(
            q, cache, out, table, cu, seq, 2, 3, rows, seq_len, causal_grid=compact
        )

    call(want, False)
    call(outs[0], True)
    spin = mfma_spinner(256)
    stream = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for out in outs:
            call(out, True)
    spin(2_000_000, stream)
    graph.replay()
    torch.cuda.synchronize()
    expected_bits = want.nan_to_num(7.0).view(torch.int32)
    for out in outs:
        assert torch.equal(out.nan_to_num(7.0).view(torch.int32), expected_bits)


@pytest.mark.parametrize('rows,prefix,compact', [
    (8192, 0, False), (12288, 0, False), (12288, 16384, False),
    (16383, 0, True), (16384, 0, True), (16384, 16384, True),
    (16384, 49152, True), (16384, 49153, False),
    (16385, 0, False), (20480, 0, False),
    (32768, 0, True), (32768, 1, False),
])
@torch.inference_mode()
def test_causal_dispatch(rows, prefix, compact, monkeypatch):
    """Keep measured query-tile tiers and both ends of the context-budget gate."""
    require_gfx950()
    seq_len = rows + prefix
    q, cache, table, cu, seq = make_case([rows], [seq_len], 1)
    out = nan_like(1, rows, cdiv(seq_len, BLK))
    names = []
    monkeypatch.setattr(host, '_run_compiled', lambda exe, *args: names.append(exe.kernel_name))
    host.nvfp4_index_score_prefill(q, cache, out, table, cu, seq, 0, 1,
        rows, seq_len, target_workgroups=4096, causal_grid=True)
    assert len(names) == 1
    assert ('_causal_' in names[0]) == compact, names
