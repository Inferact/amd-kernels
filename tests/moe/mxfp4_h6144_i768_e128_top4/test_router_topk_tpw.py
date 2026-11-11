# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The split-K top-k with two tokens per wave (``tpw`` 2, served from ``router_topk.TPW2_FROM`` tokens) against one
token per wave (AITER's ids and weights) bit for bit: random partials, rows of exact ties (resolved in the DPP tree's
order), odd token counts, every split count of the entry's routers, graph replay."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import router_topk as rt

from .workload import E, TOPK, require_gfx950


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def _partials(n, s, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    p = torch.randn((s, n, E), device="cuda", generator=g)
    # exact ties: a whole row, a block of experts, repeated values spread over both halves of a wave's lanes
    p[:, 0] = 0.25
    if n > 1:
        p[:, 1, 3:40] = 0.5
    if n > 2:
        p[:, 2] = p[:, 2, :8].repeat(1, E // 8)
    if n > 3:
        p[:, n - 1, ::5] = 1.0
    return p.permute(1, 0, 2)  # [n, s, E] view of [s, n, E], as the routers write


def _run(tpw, parts, bias, wt, ids):
    n, s, _ = parts.shape
    end = (n - 1) * parts.stride(0) + (s - 1) * parts.stride(1) + E
    rt._run_compiled(
        rt._get_kernel(E, TOPK, s, tpw),
        parts.data_ptr(),
        bias.data_ptr(),
        wt.data_ptr(),
        ids.data_ptr(),
        end * 4,
        n,
        parts.stride(0),
        parts.stride(1),
        ids.stride(0),
        0x40000000,  # 2.0
        torch.cuda.current_stream(),
    )


@pytest.mark.parametrize(
    "n,s", [(1, 16), (2, 16), (3, 48), (2047, 16), (2048, 16), (2049, 16), (4095, 8),
            (4097, 4), (8193, 4), (16385, 4), (24577, 2), (32768, 2),
            (2049, 12), (4097, 6), (8193, 3), (16385, 3), (24577, 1), (32768, 1)]
)
def test_two_tokens_per_wave_bitwise(n, s):
    parts = _partials(n, s, n)
    bias = torch.randn((E,), device="cuda", generator=torch.Generator(device="cuda").manual_seed(1)) * 0.1
    bias[::7] = 0.0
    out = []
    for tpw in (1, 2):
        wt = torch.full((n, TOPK), float("nan"), device="cuda")
        ids = torch.full((n, TOPK), -1, dtype=torch.int32, device="cuda")
        _run(tpw, parts, bias, wt, ids)
        out.append((wt, ids))
    torch.cuda.synchronize()
    (w1, i1), (w2, i2) = out
    assert torch.equal(i1, i2)
    assert torch.equal(w1.view(torch.int32), w2.view(torch.int32))
    assert bool((i1 >= 0).all()) and bool(torch.isfinite(w1).all())


def test_public_entry_switches_at_tpw2_from():
    """``splitk_biased_topk`` serves tpw 2 from TPW2_FROM rows, inside a graph too:
    replay with changed partials equals eager tpw 1, with no extra writes past M."""
    n, s = rt.TPW2_FROM + 1, 16
    bias = torch.zeros((E,), device="cuda")
    parts = _partials(n, s, 5)
    wt = torch.full((n + 1, TOPK), float("nan"), device="cuda")
    ids = torch.full((n + 1, TOPK), -1, dtype=torch.int32, device="cuda")
    rt.splitk_biased_topk(parts, bias, wt[:n], ids[:n], 2.0)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        rt.splitk_biased_topk(parts, bias, wt[:n], ids[:n], 2.0)
    parts.copy_(_partials(n, s, 6))
    g.replay()
    w1 = torch.empty((n, TOPK), device="cuda")
    i1 = torch.empty((n, TOPK), dtype=torch.int32, device="cuda")
    _run(1, parts, bias, w1, i1)
    torch.cuda.synchronize()
    assert torch.equal(ids[:n], i1) and torch.equal(wt[:n].view(torch.int32), w1.view(torch.int32))
    assert bool(torch.isnan(wt[n]).all()) and bool((ids[n] == -1).all())
