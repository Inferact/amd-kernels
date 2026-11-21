# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``topk_rows`` against ``reference.py`` bit for bit (ragged windows, ks != 0, ties, retry / fallback paths), against AITER's
stable top-k, and run to run."""

import pytest
import torch

from amd_kernels.indexer.mxfp4_prefill_topk_d128_h32_top2048 import host, reference

from .inputs import make_logits, make_windows, require_gfx950


def run(logits, ks, ke):
    out = torch.full((logits.shape[0], 2048), -7, dtype=torch.int32, device="cuda")
    host.topk_rows(logits, ks, ke, out)
    return out


CASES = [
    ("gauss", 24, 60000, "causal"),
    ("gauss", 40, 30000, "requests"),
    ("gauss", 65, 20000, "mixed"),
    ("ties", 16, 70000, "causal"),
    ("ties", 30, 20000, "requests"),
    ("negative", 16, 50000, "requests"),
    ("expo", 16, 50000, "causal"),
    ("recency", 16, 50000, "causal"),
    ("sink", 16, 50000, "causal"),
    ("spikes", 16, 50000, "causal"),
    ("blocky", 24, 50000, "causal"),
    ("zeros", 6, 30000, "causal"),
    ("gauss", 5, 9000, "causal"),
    ("gauss", 3, 700, "causal"),
]


@pytest.mark.parametrize("kind,rows,width,windows", CASES)
def test_matches_reference(kind, rows, width, windows):
    require_gfx950()
    logits = make_logits(kind, rows, width)
    ks, ke = make_windows(rows, width, windows)
    got = run(logits, ks, ke)
    want = reference.reference_topk(logits, ks, ke)
    assert torch.equal(got, want)


def test_positions_outside_windows_are_ignored():
    require_gfx950()
    rows, width = 20, 40000
    logits = make_logits("gauss", rows, width)
    ks, ke = make_windows(rows, width, "requests")
    want = reference.reference_topk(logits, ks, ke)
    pos = torch.arange(width, device="cuda")[None, :]
    outside = (pos < ks[:, None]) | (pos >= ke[:, None])
    for junk in (float("nan"), float("inf"), -float("inf"), 1e30):
        dirty = logits.masked_fill(outside, junk)
        assert torch.equal(run(dirty, ks, ke), want)


def test_infinite_logits():
    require_gfx950()
    rows, width = 12, 30000
    logits = make_logits("gauss", rows, width)
    ks, ke = make_windows(rows, width, "causal")
    logits[:, 100:140] = float("inf")  # 40 tied +inf
    logits[:, 500:900] = -float("inf")
    assert torch.equal(run(logits, ks, ke), reference.reference_topk(logits, ks, ke))


def test_row_stride_larger_than_width():
    require_gfx950()
    rows, width = 12, 25000
    wide = torch.randn(rows, width + 4096, device="cuda")
    logits = wide[:, :width]
    ks, ke = make_windows(rows, width, "causal")
    assert torch.equal(run(logits, ks, ke), reference.reference_topk(logits, ks, ke))


def test_row_order_does_not_change_the_result():
    require_gfx950()
    rows, width = 40, 40000
    logits = make_logits("gauss", rows, width)
    ks, ke = make_windows(rows, width, "requests")
    order = torch.argsort(ke - ks, descending=True, stable=True).int()
    out = torch.full((rows, 2048), -7, dtype=torch.int32, device="cuda")
    host.topk_rows(logits, ks, ke, out, order)
    assert torch.equal(out, reference.reference_topk(logits, ks, ke))


def test_deterministic():
    require_gfx950()
    logits = make_logits("ties", 33, 45000)
    ks, ke = make_windows(33, 45000, "requests")
    first = run(logits, ks, ke)
    for _ in range(3):
        assert torch.equal(run(logits, ks, ke), first)


@pytest.mark.parametrize("kind", ["gauss", "ties"])
def test_matches_aiter_stable(kind):
    require_gfx950()
    from aiter.ops.topk import top_k_per_row_prefill

    rows, width = 32, 60000
    logits = make_logits(kind, rows, width)
    ks, ke = make_windows(rows, width, "requests")
    stock = torch.full((rows, 2048), -1, dtype=torch.int32, device="cuda")
    top_k_per_row_prefill(
        logits=logits, rowStarts=ks, rowEnds=ke, indices=stock, values=None, numRows=rows,
        stride0=logits.stride(0), stride1=1, k=2048, stable=True,
    )  # fmt: skip
    stock = torch.where(stock >= 0, stock - ks[:, None], -1)
    assert torch.equal(run(logits, ks, ke), stock)
