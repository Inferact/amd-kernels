# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``fp8_sparse_mla_decode`` against ``reference.py``, with bf16 q and with the pre-quantized q (``q8``).

Served shapes: rows 12 / 24 / 40 / 65 / 80 / 128 / 192 of 4-row requests with ~256 keys per row, the CUDA-graph padding
sizes, the CUDA-graph padding rows (empty slices), key counts around every tile edge and up to 2048, a strided q and out,
extreme per-block scales and logits. Unreferenced cache slots are NaN. Every case: against the fp32 reference, two
launches bit for bit equal, the pre-quantized path bit for bit equal to the bf16 path, a CUDA graph replayed into
NaN-filled outputs.
"""

import pytest
import torch
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import (
    fp8_sparse_mla_decode,
    supports_fp8_sparse_mla_decode,
)
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import (
    H,
    L,
    check_close,
    quantize_q,
    reference,
)

from .cases import GRAPH_ROWS, SERVED_ROWS, SM_SCALE, make_batch, require_gfx950, served_counts


def run_case(counts, seed, q_pad=0, out_pad=0, q_scale=0.5, exp_range=(-4, 4)):
    gen = torch.Generator().manual_seed(seed)
    q, cache, indptr, slots = make_batch(counts, gen, q_scale=q_scale, exp_range=exp_range)
    rows = len(counts)
    ref_out, ref_lse = reference(q, cache, indptr, slots, SM_SCALE)
    q8 = quantize_q(q)
    if q_pad:
        big = torch.zeros(rows, H, q.shape[2] + q_pad, dtype=q.dtype, device=q.device)
        big[:, :, : q.shape[2]] = q
        q = big[:, :, : q.shape[2]]
        big8 = torch.zeros(rows, H, q8.shape[2] + q_pad * 2, dtype=q8.dtype, device=q.device)
        big8[:, :, : q8.shape[2]] = q8
        q8 = big8[:, :, : q8.shape[2]]
    out = None
    if out_pad:
        out = torch.full((rows, H, L + out_pad), float("nan"), dtype=torch.bfloat16, device=q.device)[:, :, :L]
    assert supports_fp8_sparse_mla_decode(q, cache, indptr, slots, out)
    assert supports_fp8_sparse_mla_decode(None, cache, indptr, slots, out, q8=q8)
    o1, l1 = fp8_sparse_mla_decode(q, cache, indptr, slots, SM_SCALE, out)
    o2, l2 = fp8_sparse_mla_decode(q, cache, indptr, slots, SM_SCALE)
    o3, l3 = fp8_sparse_mla_decode(None, cache, indptr, slots, SM_SCALE, q8=q8)
    torch.cuda.synchronize()
    check_close(o1.cpu(), l1.cpu(), ref_out.cpu(), ref_lse.cpu())
    assert torch.equal(o1, o2) and torch.equal(l1, l2)
    assert torch.equal(o3, o2) and torch.equal(l3, l2), "pre-quantized q differs from the in-kernel quantization"
    og = torch.full_like(o2, float("nan"))
    lg = torch.full_like(l2, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout, glse = fp8_sparse_mla_decode(q, cache, indptr, slots, SM_SCALE)
    og.copy_(gout), lg.copy_(glse)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(gout, o2) and torch.equal(glse, l2)


@pytest.mark.parametrize("rows", sorted(set(SERVED_ROWS) | set(GRAPH_ROWS)))
@torch.inference_mode()
def test_served_rows(rows):
    require_gfx950()
    gen = torch.Generator().manual_seed(rows)
    run_case(served_counts(rows, gen), rows)


@pytest.mark.parametrize("rows,pad", [(12, 3), (40, 15), (80, 15), (192, 63)])
@torch.inference_mode()
def test_padding_rows(rows, pad):
    require_gfx950()
    gen = torch.Generator().manual_seed(rows + 1)
    run_case(served_counts(rows, gen, pad=pad), rows + 1)


@pytest.mark.parametrize(
    "counts",
    [
        [1, 2, 3, 0],
        [31, 32, 33, 63],
        [64, 65, 95, 96],
        [127, 128, 129, 191],
        [192, 255, 256, 257],
        [271, 272, 273, 287],
        [288, 289, 300, 319],
        [320, 321, 383, 384],
        [512, 576, 1000, 2047],
        [2048, 2048, 2048, 2048],
        [0, 0, 0, 0],
    ],
    ids=lambda c: "_".join(map(str, c)),
)
@torch.inference_mode()
def test_key_counts(counts):
    require_gfx950()
    run_case(counts, sum(counts) + 5)


@pytest.mark.parametrize("rows", [24, 42, 43, 80])
@torch.inference_mode()
def test_key_counts_across_tiers(rows):
    require_gfx950()
    gen = torch.Generator().manual_seed(rows + 11)
    counts = [int(x) for x in torch.randint(0, 340, (rows,), generator=gen)]
    run_case(counts, rows + 11)


@torch.inference_mode()
def test_strided_q_and_out():
    require_gfx950()
    gen = torch.Generator().manual_seed(7)
    run_case(served_counts(24, gen), 7, q_pad=64, out_pad=64)


@pytest.mark.parametrize("rows", [12, 80])
@torch.inference_mode()
def test_extreme_scales_and_logits(rows):
    """Per-block scales from 2^-20 to 2^7 and logits spanning tens of units (the running max moves between tiles)."""
    require_gfx950()
    gen = torch.Generator().manual_seed(rows + 21)
    run_case(served_counts(rows, gen), rows + 21, q_scale=8.0, exp_range=(-20, 8))


@pytest.mark.parametrize("rows,pad", [(40, 40), (40, 8), (12, 3), (65, 65), (80, 15), (192, 63)])
@torch.inference_mode()
def test_garbage_past_csr_end(rows, pad):
    """vLLM's slot buffer is ``rows * 2048`` long and only ``[0, nnz)`` is written: the entries past the CSR end are
    stale or uninitialized. Empty rows (CUDA-graph padding, or every row of a rank that holds none of the keys) must
    not read a record through them (the served model faulted on an all-empty rank before the first-slot guard)."""
    require_gfx950()
    gen = torch.Generator().manual_seed(1000 + rows + pad)
    counts = served_counts(rows, gen, pad=pad)
    q, cache, indptr, slots = make_batch(counts, gen)
    ref_out, ref_lse = reference(q, cache, indptr, slots, SM_SCALE)
    big = torch.full((rows * 2048,), 0x7FFFFFF0, dtype=torch.int32, device=slots.device)
    big[: slots.numel()] = slots
    o, l = fp8_sparse_mla_decode(q, cache, indptr, big, SM_SCALE)
    torch.cuda.synchronize()
    check_close(o.cpu(), l.cpu(), ref_out.cpu(), ref_lse.cpu())
