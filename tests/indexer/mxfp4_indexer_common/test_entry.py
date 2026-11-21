# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Each indexer entry against the direct shape-package call on the same operands, bit for bit, at the smallest and
largest served row counts; the reasons are None there; the scratch helpers act on the routed package's scratch."""

import importlib

import pytest
import torch

from amd_kernels.indexer.mxfp4_indexer_common import (
    mxfp4_paged_score_topk,
    mxfp4_paged_score_topk_unsupported_reason,
    mxfp4_prefill_score,
    mxfp4_prefill_score_unsupported_reason,
    mxfp4_prefill_topk,
    mxfp4_prefill_topk_unsupported_reason,
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

PAGED = importlib.import_module(paged._SHAPE_HOSTS[(32, 128, 2048, 64)])
SCORE = importlib.import_module(prefill._SCORE_HOSTS[(32, 128)])
TOPK = importlib.import_module(prefill._TOPK_HOSTS[(2048,)])


def outputs(rows, topk=2048):
    return (torch.full((rows, topk), -7, dtype=torch.int32, device="cuda"),
            torch.full((rows, topk), 3.0, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize("seqs,qlen,width", [(1, 1, 4096), (2, 6, 94208), (64, 4, 8192), (256, 1, 16384)])
def test_paged_equals_package(seqs, qlen, width):
    require_gfx950()
    inputs = paged_inputs(seqs, qlen, width, seed=seqs * 7 + width)
    rows = seqs * qlen
    got, want = outputs(rows), outputs(rows)
    assert mxfp4_paged_score_topk_unsupported_reason(**inputs, out_indices=got[0], out_scores=got[1]) is None
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
    assert PAGED.host._scratch_set.cache_info().currsize >= 2
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


@pytest.mark.parametrize("rows,keys", [(1, 64), (8, 4096), (512, 32768)])
def test_prefill_score_equals_package(rows, keys):
    require_gfx950()
    args = score_inputs(rows, keys, seed=rows + keys)
    width = args[3].shape[0] * 64
    got = torch.full((rows, width), 7.0, device="cuda")
    want = got.clone()
    assert mxfp4_prefill_score_unsupported_reason(*args, got, 0.5) is None
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
    assert mxfp4_prefill_topk_unsupported_reason(logits, ks, ke, got, perm) is None
    mxfp4_prefill_topk(logits, ks, ke, got, perm)
    TOPK.topk_rows(logits, ks, ke, want, perm)
    torch.cuda.synchronize()
    assert torch.equal(got, want)
    assert prefill_topk_workspace_bytes(rows, width, topk=2048) == TOPK.workspace_bytes(rows, width)
