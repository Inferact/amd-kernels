# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``score_rows`` against ``reference.py`` (rel. 1e-5 of the row max), against AITER's scorer (2 ulp), run to run."""

import pytest
import torch
from amd_kernels.indexer.mxfp4_prefill_score_d128_h32 import host, reference

from .inputs import make_inputs, require_gfx950

SENTINEL = float("nan")
SHAPES = [
    (1, 64), (7, 100), (8, 513), (37, 5000), (100, 30000), (257, 9000),
]  # fmt: skip


def run(inputs, weight_scale=0.37, width=None):
    q_payload, q_scale, weights, key_payload, key_scale, ks, ke = inputs
    width = width or key_payload.shape[0] * 64
    out = torch.full((q_payload.shape[0], width), SENTINEL, dtype=torch.float32, device="cuda")
    host.score_rows(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out, weight_scale)
    return out


def window(inputs, width):
    ks, ke = inputs[5], inputs[6]
    pos = torch.arange(width, device="cuda")[None, :]
    return (pos >= ks[:, None]) & (pos < ke[:, None])


@pytest.mark.parametrize("rows,keys", SHAPES)
def test_matches_reference(rows, keys):
    require_gfx950()
    inputs = make_inputs(rows, keys)
    got = run(inputs)
    want = reference.reference_logits(*inputs, 0.37)
    mask = window(inputs, got.shape[1])
    assert not torch.isnan(got[mask]).any()
    scale = want[mask].abs().max()
    assert (got[mask] - want[mask]).abs().max() <= 1e-5 * scale


def test_two_requests_and_empty_rows():
    require_gfx950()
    rows, keys = 61, 12000
    windows = [(0, 7000 + r) if r < 30 else (4096 + 3, keys - 3000 + r) for r in range(rows)]
    windows[5] = (100, 100)
    windows[40] = (0, 0)
    inputs = make_inputs(rows, keys, windows)
    got = run(inputs)
    want = reference.reference_logits(*inputs, 0.37)
    mask = window(inputs, got.shape[1])
    assert (got[mask] - want[mask]).abs().max() <= 1e-5 * want[mask].abs().max()


def test_matches_aiter_scorer():
    require_gfx950()
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    rows, keys = 300, 20000
    inputs = make_inputs(rows, keys)
    q_payload, q_scale, weights, key_payload, key_scale, ks, ke = inputs
    pages = key_payload.shape[0]
    table = torch.arange(pages + 8, dtype=torch.int32, device="cuda").unsqueeze(0)
    batches = torch.zeros(rows, dtype=torch.int32, device="cuda")
    _, schedule, n_ctas = compute_prefill_schedule(batches, ks, ke, 256, 4096, pages * 64)
    want = torch.full((rows, pages * 64), SENTINEL, dtype=torch.float32, device="cuda")
    flydsl_pa_mqa_logits_fp4_prefill(
        q_payload, q_scale, key_payload, key_scale, table, weights, batches, ks, ke, pages * 64,
        weight_scale=0.37, cta_info=schedule, n_ctas=n_ctas, out=want,
    )  # fmt: skip
    got = run(inputs)
    mask = window(inputs, got.shape[1])
    ulp = torch.finfo(torch.float32).eps * want[mask].abs().max()
    assert (got[mask] - want[mask]).abs().max() <= 2 * ulp


def test_deterministic():
    require_gfx950()
    inputs = make_inputs(100, 30000)
    a, b = run(inputs), run(inputs)
    assert torch.equal(torch.nan_to_num(a, nan=-1.0), torch.nan_to_num(b, nan=-1.0))
