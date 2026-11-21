# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Test inputs: MXFP4 index caches, block tables, quantized query rows, causal row bounds."""

import pytest
import torch

HEADS = 32
PAGE = 64


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def make_inputs(
    num_seqs: int,
    query_len: int,
    width: int,
    lengths=None,
    seed: int = 0,
    page_stride: int = 4352,
    hot=(),
    scale_range=(118, 124),
    positive_weights: bool = False,
):
    """Random operands for ``num_seqs`` sequences of ``query_len`` rows against ``width`` keys a sequence (a multiple
    of 64): ``lengths`` the sequences' key counts (default: random in [width / 2, width]), row ``r`` of a sequence
    bounded at ``length - (query_len - 1 - r)``. Key bytes, scales and query bytes are random (every byte is a valid
    e2m1 pair / e8m0 scale); ``hot`` is ``(first key, last key, scale increase)`` runs whose keys are scored higher.
    Returns a dict of the entry's operands."""
    assert width % PAGE == 0
    gen = torch.Generator(device="cuda").manual_seed(seed)
    blocks = width // PAGE
    pages = num_seqs * blocks
    rows = num_seqs * query_len
    table = torch.randperm(pages, generator=gen, device="cuda", dtype=torch.int32).reshape(num_seqs, blocks)
    flat = torch.zeros(pages, page_stride, dtype=torch.uint8, device="cuda")
    flat[:, :4096] = torch.randint(0, 256, (pages, 4096), dtype=torch.uint8, device="cuda", generator=gen)
    lo, hi = scale_range
    flat[:, 4096:4352] = torch.randint(lo, hi, (pages, 256), dtype=torch.uint8, device="cuda", generator=gen)
    for first, last, add in hot:
        for s in range(num_seqs):
            flat[table[s, first // PAGE : last // PAGE].long(), 4096:4352] = lo + add
    cache = flat[:, :4352].unflatten(1, (PAGE, 68))
    q_payload = torch.randint(0, 256, (rows, HEADS, 64), dtype=torch.uint8, device="cuda", generator=gen)
    q_scales = torch.zeros(rows, 1, 4, 16, 4, dtype=torch.uint8, device="cuda")
    q_scales[..., :2] = torch.randint(lo, hi, (rows, 1, 4, 16, 2), dtype=torch.uint8, device="cuda", generator=gen)
    weights = torch.randn(rows, HEADS, device="cuda", generator=gen) * 0.2
    if positive_weights:
        weights = weights.abs()
    if lengths is None:
        lengths = torch.randint(width // 2, width + 1, (num_seqs,), device="cuda", generator=gen).tolist()
    r = torch.arange(query_len, device="cuda")
    ends = (torch.tensor(lengths, device="cuda")[:, None] - (query_len - 1 - r)[None, :]).clamp(min=0).reshape(-1)
    return dict(
        q_payload=q_payload,
        q_scales=q_scales,
        weights=weights,
        cache=cache,
        block_table=table,
        query_start_loc=torch.arange(num_seqs + 1, dtype=torch.int32, device="cuda") * query_len,
        row_ends=ends.int(),
        max_query_len=query_len,
    )
