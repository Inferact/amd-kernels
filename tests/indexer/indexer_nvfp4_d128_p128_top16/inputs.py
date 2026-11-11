# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Test inputs: NVFP4 index caches, block tables, index queries, score rows."""

import math

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK, D, nvfp4_index_pages


def require_gfx950():
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("requires gfx950")


def index_cache(num_pages: int, page_step: int = 1, saturate: bool = False):
    """A filled NVFP4 index cache ``[num_pages, 128, 72]``; ``page_step`` > 1 gives a
    view of a larger cache whose pages sit ``page_step`` pages apart (``stride(0) =
    page_step * 9216`` bytes). ``saturate``: a tenth of the tokens scaled past
    448, so the fp8 route clamps their keys."""
    total = num_pages * page_step
    cache = torch.empty(total, BLK, 72, dtype=torch.uint8, device="cuda")
    step = 512  # pages per chunk
    for p0 in range(0, total, step):
        n = min(step, total - p0)
        keys = torch.randn(n * BLK, D, device="cuda") * 2
        if saturate:
            big = torch.rand(n * BLK, 1, device="cuda") < 0.1
            keys = torch.where(big, keys * 400, keys)
        cache[p0 : p0 + n] = nvfp4_index_pages(keys.bfloat16())
    return cache[::page_step]


def block_table(seq_lens, share_prefixes: bool = False):
    """(table [reqs, max_blocks] int32, pages needed): each request's blocks on
    distinct pages in random order; ``share_prefixes``: even requests share
    request 0's first <= 10 blocks, odd ones request 3 (r // 6)'s first <= 6."""
    widths = [math.ceil(s / BLK) for s in seq_lens]
    num_pages = sum(widths) + 2
    perm = torch.randperm(num_pages, device="cuda", dtype=torch.int32)
    table = torch.zeros(len(seq_lens), max(max(widths), 1), dtype=torch.int32, device="cuda")
    start = 0
    for r, w in enumerate(widths):
        table[r, :w] = perm[start : start + w]
        start += w
    if share_prefixes:
        for r in range(len(seq_lens)):
            src = 0 if r % 2 == 0 else 3 * (r // 6)
            n = min(widths[r], widths[src], 10 if r % 2 == 0 else 6)
            if r != src:
                table[r, :n] = table[src, :n]
    return table, num_pages


def queries(rows: int, heads: int) -> torch.Tensor:
    return (torch.randn(rows, heads, D, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def decode_lens(num_reqs: int, hi: int, seed: int) -> list[int]:
    """Random lengths in [1, hi) with the edge lengths (0 = graph padding, 1-3 =
    shorter than the query, 127 / 128 / 129, 1024) on the odd requests."""
    gen = torch.Generator().manual_seed(seed)
    lens = torch.randint(1, hi, (num_reqs,), generator=gen).tolist()
    special = [0, 1, 2, 3, 127, 128, 129, 1024]
    for i, v in enumerate(special[: num_reqs // 2]):
        lens[2 * i + 1] = v
    return lens


def nan_like(*shape):
    return torch.full(shape, float("nan"), device="cuda")


def assert_scores(got, want, rtol, atol, msg=None):
    """Sentinels and -inf bit for bit; finite scores to rtol / atol. Returns
    (max relative error, relative L2 error, cosine) of the finite scores."""
    exact_slots = ~torch.isfinite(want) | (want >= 1e28)
    assert torch.equal(got[exact_slots], want[exact_slots].to(got.dtype)), msg
    g, w = got[~exact_slots].double(), want[~exact_slots].double()
    torch.testing.assert_close(g, w, rtol=rtol, atol=atol, msg=msg)
    if g.numel() == 0:
        return 0.0, 0.0, 1.0
    rel = ((g - w).abs() / w.abs().clamp(min=1.0)).max().item()
    l2 = ((g - w).norm() / w.norm().clamp(min=1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(g, w, dim=0).item()
    return rel, l2, cos
