# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The routing sort + tile map + sorted scale tiles in one launch (``common.sort_scales_all``) against the routing, the
``sort_count_all`` + scale sort pair (padded totals, block experts, per-expert (row id, weight) sets, tile map) and the
torch gather + AITER shuffle, on uniform, skewed, 4-expert and duplicate-id routing, odd token counts, every
tokens-per-thread tier and both block sizes; rows come out in a fixed order (two runs, identical bits)."""

import pytest
import torch
from aiter.utility.fp4_utils import e8m0_shuffle
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.scale_sort_tile_map import scale_sort_tile_map
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort import SortBuffers
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort_count_all import moe_sort_count_all
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort_scales_all import MAX_TOKENS, moe_sort_scales_all
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.tile_map import tile_map_grid

from .workload import E, H, I, TOPK, make_routing, require_gfx950

TOKENS = [1, 7, 513, 777, 1024, 1025, 2047, 2048, 3001, 4096, 8192]


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def routing(n, kind):
    if kind == "four_experts":
        w, _ = make_routing(n, seed=n)
        return w, torch.arange(TOPK, dtype=torch.int32, device="cuda").expand(n, TOPK).contiguous()
    if kind == "duplicates":  # two slots of a token on one expert
        w, ids = make_routing(n, seed=n + 5)
        ids[:, 1] = ids[:, 0]
        ids[: n // 2, 3] = 9
        return w, ids.contiguous()
    return make_routing(n, seed=n + 3, skewed=kind == "skewed")


def scales(n):
    g = torch.Generator(device="cuda").manual_seed(n)
    return torch.randint(100, 140, (n, H // 32), dtype=torch.uint8, device="cuda", generator=g)


def run(n, ids, w, xs, bm, fused):
    bufs = SortBuffers.allocate(n, E, TOPK, bm, "cuda")
    bufs.sorted_ids.fill_(-7)
    bufs.sorted_weights.fill_(float("nan"))
    bufs.sorted_expert_ids.fill_(-7)
    grid = tile_map_grid(bufs.max_sorted // bm, I)
    tm = torch.full((grid + 1,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
    a_s = torch.full(((bufs.max_sorted + 31) // 32 * 32, H // 32), 0xA5, dtype=torch.uint8, device="cuda")
    if fused:
        moe_sort_scales_all(bufs, ids, w, xs, a_s, tm, grid, n, E=E, I=I)
    else:
        moe_sort_count_all(bufs, ids, w, tm, grid, n, E=E, I=I)
        scale_sort_tile_map(
            xs, bufs.sorted_ids, bufs.sorted_expert_ids, bufs.num_valid_ids, a_s, tm, grid, n, bm,
            with_tile_map=False,
        )  # fmt: skip
    return bufs, tm, a_s


def per_expert(bufs, bm):
    nv = int(bufs.num_valid_ids[0])
    eids = bufs.sorted_expert_ids[: nv // bm].tolist()
    ids, ws = bufs.sorted_ids[:nv].tolist(), bufs.sorted_weights[:nv].tolist()
    per = {}
    for i, e in enumerate(eids):
        per.setdefault(e, []).extend(zip(ids[i * bm : (i + 1) * bm], ws[i * bm : (i + 1) * bm]))
    return {e: sorted(v) for e, v in per.items()}


@pytest.mark.parametrize("bm", [128, 256])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "four_experts", "duplicates"])
@pytest.mark.parametrize("n", TOKENS)
def test_sort_scales_all_matches_routing_and_two_launches(n, kind, bm):
    w, ids = routing(n, kind)
    xs = scales(n)
    got, tm, a_s = run(n, ids, w, xs, bm, True)
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
    # scale tiles: each valid row its token's e8m0 bytes, padding rows 0, AITER-shuffled
    gathered = torch.zeros((nv[0], H // 32), dtype=torch.uint8, device="cuda")
    gathered[~pad] = xs[tok]
    want = e8m0_shuffle(gathered).view(torch.uint8).reshape(-1)[: nv[0] * (H // 32)]
    assert torch.equal(a_s.reshape(-1)[: nv[0] * (H // 32)], want)
    # the two-launch pair: the same table, rows per expert and tile map
    ref, ref_tm, _ = run(n, ids, w, xs, bm, False)
    assert ref.num_valid_ids.tolist() == nv
    assert per_expert(got, bm) == per_expert(ref, bm)
    assert torch.equal(tm, ref_tm)
    # a fixed row order: a second run writes the same bits
    again, tm2, a_s2 = run(n, ids, w, xs, bm, True)
    assert torch.equal(again.sorted_ids[: nv[0]], got.sorted_ids[: nv[0]])
    assert torch.equal(again.sorted_weights[: nv[0]], got.sorted_weights[: nv[0]])
    assert torch.equal(a_s2.reshape(-1)[: nv[0] * (H // 32)], a_s.reshape(-1)[: nv[0] * (H // 32)])
    assert torch.equal(tm2, tm)


def test_sort_scales_all_rejects_more_tokens():
    w, ids = make_routing(MAX_TOKENS + 1, seed=1)
    with pytest.raises(ValueError):
        run(MAX_TOKENS + 1, ids, w, scales(MAX_TOKENS + 1), 256, True)
