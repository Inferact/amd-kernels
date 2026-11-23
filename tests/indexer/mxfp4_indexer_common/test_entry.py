# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Each indexer entry against the direct shape-package call on the same operands, bit for bit, at the smallest and
largest served row counts; the reasons are None there; the scratch helpers act on the routed package's scratch."""

import importlib

import pytest
import torch

from amd_kernels.indexer.mxfp4_indexer_common import (
    mxfp4_paged_score_topk,
    mxfp4_paged_score_topk_blocker,
    mxfp4_prefill_score,
    mxfp4_prefill_score_blocker,
    mxfp4_prefill_topk,
    mxfp4_prefill_topk_blocker,
    paged,
    paged_score_topk_preallocate,
    paged_score_topk_prefer,
    paged_score_topk_reset_counts,
    prefill,
    prefill_topk_workspace_bytes,
)
from tests.indexer.mxfp4_paged_score_topk_d128_h32_top2048.inputs import make_inputs as paged_inputs
from tests.indexer.mxfp4_paged_score_topk_d128_h32_top2048.inputs import require_gfx950
from tests.indexer.mxfp4_prefill_score_d128_h32.inputs import make_inputs as score_inputs
from tests.indexer.mxfp4_prefill_topk_d128_h32_top2048.inputs import make_logits, make_windows

PAGED = importlib.import_module(paged._ROUTES.hosts[(32, 128, 2048, 64)])
SCORE = importlib.import_module(prefill._SCORE_ROUTES.hosts[(32, 128)])
TOPK = importlib.import_module(prefill._TOPK_ROUTES.hosts[(2048,)])


def outputs(rows, topk=2048):
    return (torch.full((rows, topk), -7, dtype=torch.int32, device="cuda"),
            torch.full((rows, topk), 3.0, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("seqs,qlen,width", [(1, 1, 4096), (2, 6, 94208), (64, 4, 8192), (256, 1, 16384)])
def test_paged_equals_package(seqs, qlen, width):
    require_gfx950()
    inputs = paged_inputs(seqs, qlen, width, seed=seqs * 7 + width)
    rows = seqs * qlen
    got, want = outputs(rows), outputs(rows)
    assert mxfp4_paged_score_topk_blocker(**inputs, out_indices=got[0], out_scores=got[1]) is None
    mxfp4_paged_score_topk(**inputs, out_indices=got[0], out_scores=got[1])
    PAGED.mxfp4_paged_score_topk(**inputs, out_indices=want[0], out_scores=want[1])
    torch.cuda.synchronize()
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    assert paged_score_topk_prefer(seqs, width, cache=inputs["cache"], out_indices=got[0]) == PAGED.prefer_fused(
        seqs, width)


def test_paged_scratch_helpers():
    require_gfx950()
    inputs = paged_inputs(4, 6, 4096)
    out = outputs(24)[0]
    stream = torch.cuda.Stream()
    paged_score_topk_preallocate("cuda", [torch.cuda.current_stream(), stream], cache=inputs["cache"], out_indices=out)
    dev = inputs["cache"].device
    assert len(PAGED.host._SETS) >= 2
    with torch.cuda.stream(stream):
        counts = PAGED.host._scratch(dev)[4]
    counts.fill_(5)
    with torch.cuda.stream(stream):
        paged_score_topk_reset_counts(inputs["query_start_loc"], 6, cache=inputs["cache"], out_indices=out)
    torch.cuda.synchronize()
    tile = PAGED.host.plan(6, 4)[0]
    used = 4 * -(-6 // tile) * PAGED.geometry.COUNTER_BYTES // 4
    assert int(counts[:used].abs().sum()) == 0 and bool((counts[used:] == 5).all())
    counts.zero_()
    torch.cuda.synchronize()


def ragged_padded(qlens, pad, width, seed):
    """Sequences of ``qlens`` rows (the last rows of 6-row sequences) and ``pad`` padding sequences without rows."""
    base = paged_inputs(len(qlens) + pad, 6, width, seed=seed)
    idx = torch.tensor([s * 6 + r for s, q in enumerate(qlens) for r in range(6 - q, 6)], device="cuda")
    starts = torch.tensor([0, *qlens, *[0] * pad], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
    rows = {k: base[k][idx].contiguous() for k in ("q_payload", "q_scales", "weights", "row_ends")}
    return {**base, **rows, "query_start_loc": starts, "max_query_len": max(qlens)}


@pytest.mark.parametrize("batch", ["uniform", "ragged_padded"])
def test_paged_scratch_rows(batch):
    """Sets sized for 40 rows: a 40-row call (uniform, or ragged with padding sequences) equals the package's at full
    size, a 41-row call gets a reason; the release keeps only the given streams' sets."""
    require_gfx950()
    if batch == "uniform":
        inputs = paged_inputs(10, 4, 16384, seed=5)
    else:
        inputs = ragged_padded([5, 1, 6, 3, 5, 5, 5, 4, 6], 7, 16384, seed=5)
    big = paged_inputs(41, 1, 4096)
    got, want, over = outputs(40), outputs(40), outputs(41)
    stream = torch.cuda.Stream()
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            PAGED.mxfp4_paged_score_topk(**inputs, out_indices=want[0], out_scores=want[1])
        torch.cuda.synchronize()
        paged_score_topk_preallocate("cuda", [stream], cache=inputs["cache"], out_indices=got[0], rows=40,
                                     release_others=True)
        dev = torch.cuda.current_device()
        assert [k[1] for k in PAGED.host._SETS if k[0] == dev] == [stream.cuda_stream]
        assert PAGED.host._SETS[(dev, stream.cuda_stream)][0].numel() == 40 * PAGED.geometry.POOL
        assert "the scratch holds 40" in mxfp4_paged_score_topk_blocker(
            **big, out_indices=over[0], out_scores=over[1])
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            assert mxfp4_paged_score_topk_blocker(**inputs, out_indices=got[0], out_scores=got[1]) is None
            paged_score_topk_reset_counts(inputs["query_start_loc"], inputs["max_query_len"], cache=inputs["cache"],
                                          out_indices=got[0])
            mxfp4_paged_score_topk(**inputs, out_indices=got[0], out_scores=got[1])
        torch.cuda.synchronize()
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    finally:
        torch.cuda.synchronize()
        PAGED.host.set_scratch_rows(PAGED.geometry.MAX_ROWS)
        PAGED.host.release_scratch("cuda")


@pytest.mark.parametrize("rows,keys", [(1, 64), (8, 4096), (512, 32768)])
def test_prefill_score_equals_package(rows, keys):
    require_gfx950()
    args = score_inputs(rows, keys, seed=rows + keys)
    width = args[3].shape[0] * 64
    got = torch.full((rows, width), 7.0, device="cuda")
    want = got.clone()
    assert mxfp4_prefill_score_blocker(*args, got, 0.5) is None
    mxfp4_prefill_score(*args, got, 0.5)
    SCORE.score_rows(*args, want, 0.5)
    torch.cuda.synchronize()
    assert torch.equal(got, want)


@pytest.mark.parametrize("kind,rows,width,windows,order", [
    ("gauss", 1, 700, "causal", False),
    ("ties", 16, 70000, "causal", True),
    ("gauss", 65, 20000, "mixed", False),
    ("gauss", 256, 60000, "requests", True),
])
def test_prefill_topk_equals_package(kind, rows, width, windows, order):
    require_gfx950()
    logits = make_logits(kind, rows, width, seed=rows)
    ks, ke = make_windows(rows, width, windows)
    perm = torch.argsort(ke - ks, descending=True).int() if order else None
    got = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    want = got.clone()
    assert mxfp4_prefill_topk_blocker(logits, ks, ke, got, perm) is None
    mxfp4_prefill_topk(logits, ks, ke, got, perm)
    TOPK.topk_rows(logits, ks, ke, want, perm)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    assert prefill_topk_workspace_bytes(rows, width, topk=2048) == TOPK.workspace_bytes(rows, width)
