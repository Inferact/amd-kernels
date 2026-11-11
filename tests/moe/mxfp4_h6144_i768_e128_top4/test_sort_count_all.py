# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The one-launch routing sort + tile map (``common.sort_count_all``: every CTA counts all pairs) against the routing
(padded totals, block experts, per-expert (row id, weight) sets) and the torch tile map, on uniform, skewed and
4-expert routing; then the scale sort without tile-map CTAs after it, against the torch gather + AITER shuffle."""

import pytest
import torch
from aiter.utility.fp4_utils import e8m0_shuffle
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.scale_sort_tile_map import scale_sort_tile_map
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort import SortBuffers
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort_count_all import moe_sort_count_all
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.tile_map import tile_map_grid

from .workload import E, H, I, TOPK, make_routing, require_gfx950, tile_map_ref

TOKENS = [1, 7, 513, 777, 1024, 2047, 2048, 3001, 4096]


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def routing(n, kind):
    if kind == "four_experts":
        w, _ = make_routing(n, seed=n)
        ids = torch.arange(TOPK, dtype=torch.int32, device="cuda").expand(n, TOPK).contiguous()
        return w, ids
    return make_routing(n, seed=n + 3, skewed=kind == "skewed")


def run(n, ids, w, bm):
    bufs = SortBuffers.allocate(n, E, TOPK, bm, "cuda")
    bufs.sorted_ids.fill_(-7)
    bufs.sorted_weights.fill_(float("nan"))
    bufs.sorted_expert_ids.fill_(-7)
    grid = tile_map_grid(bufs.max_sorted // bm, I)
    tm = torch.full((grid + 1,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
    moe_sort_count_all(bufs, ids, w, tm, grid, n, E=E, I=I)
    return bufs, tm, grid


@pytest.mark.parametrize("bm", [128, 256])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "four_experts"])
@pytest.mark.parametrize("n", TOKENS)
def test_sort_count_all_matches_routing_and_tile_map(n, kind, bm):
    w, ids = routing(n, kind)
    got, tm, grid = run(n, ids, w, bm)
    counts = torch.bincount(ids.flatten().long(), minlength=E)
    padded = (counts + bm - 1) // bm * bm
    starts = torch.cumsum(padded, 0) - padded
    nv = got.num_valid_ids.tolist()
    assert nv[0] == int(padded.sum()) and nv[1] == int(starts[-1])
    blocks = nv[0] // bm
    expect_eid = torch.repeat_interleave(torch.arange(E, device="cuda"), padded // bm)
    assert torch.equal(got.sorted_expert_ids[:blocks].long(), expect_eid)
    sid = got.sorted_ids[: nv[0]].long()
    sw = got.sorted_weights[: nv[0]]
    pad = sid == n
    assert torch.equal(sw[pad], torch.zeros_like(sw[pad]))
    tok, slot = sid[~pad] & 0xFFFFFF, sid[~pad] >> 24
    row_expert = torch.repeat_interleave(torch.arange(E, device="cuda"), padded)[~pad]
    pair = tok * TOPK + slot
    assert torch.equal(torch.sort(pair).values, torch.arange(n * TOPK, device="cuda"))
    assert torch.equal(ids.flatten().long()[pair], row_expert)
    assert torch.equal(w.flatten()[pair], sw[~pad])
    within = torch.arange(nv[0], device="cuda") - torch.repeat_interleave(starts, padded)
    assert torch.equal(pad, within >= torch.repeat_interleave(counts, padded))
    ref_tm = tile_map_ref(got.sorted_expert_ids, got.num_valid_ids, grid, bm)
    assert torch.equal(tm, ref_tm)
    # the scale sort without tile-map CTAs: each valid row its token's e8m0 bytes, padding rows 0, AITER-shuffled; the
    # tile map untouched
    g = torch.Generator(device="cuda").manual_seed(n)
    xs = torch.randint(100, 140, (n, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    a_s = torch.full(((got.max_sorted + 31) // 32 * 32, H // 32), 0xA5, dtype=torch.uint8, device="cuda")
    scale_sort_tile_map(
        xs, got.sorted_ids, got.sorted_expert_ids, got.num_valid_ids, a_s, tm, grid, n, bm, with_tile_map=False
    )  # fmt: skip
    gathered = torch.zeros((nv[0], H // 32), dtype=torch.uint8, device="cuda")
    gathered[~pad] = xs[tok]
    want = e8m0_shuffle(gathered).view(torch.uint8).reshape(-1)[: nv[0] * (H // 32)]
    assert torch.equal(a_s.reshape(-1)[: nv[0] * (H // 32)], want)
    assert torch.equal(tm, ref_tm)
