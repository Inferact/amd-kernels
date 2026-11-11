# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""CTA-major histogram sort + exact GEMM1 tile map (``common.sort_tile_map``): the workspace against per-CTA torch
histograms, row / weight sets against the routing, the map against the torch tile map, gathered scales
against torch + AITER, every writable buffer guarded, eager and replayed."""
import pytest
import torch
from aiter.utility.fp4_utils import e8m0_shuffle
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.scale_sort_tile_map import scale_sort_tile_map
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort import SORT_CTAS, SortBuffers
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort_tile_map import moe_sort_tile_map
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.tile_map import tile_map_grid
from .test_sort_scales_all import routing, scales
from .workload import E, TOPK, H, I, require_gfx950, tile_map_ref


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


@pytest.mark.parametrize('n', [8192, 8193, 12289, 16384, 16385, 32767, 32768, 65536, 87381])
@pytest.mark.parametrize('kind', ['uniform', 'skewed', 'four_experts', 'duplicates'])
def test_cta_histogram_sort_and_map(n, kind):
    bm = 256
    weights, ids = routing(n, kind)
    xs = scales(n)
    b = SortBuffers.allocate(n, E, TOPK, bm, 'cuda')
    grid = tile_map_grid(b.max_sorted // bm, I)
    guards = []

    def guarded(size, dtype):
        whole = torch.full((size + 128,), 37, dtype=dtype, device='cuda')
        guards.append(whole)
        return whole[64:-64]

    for name in ['sorted_ids', 'sorted_expert_ids', 'num_valid_ids', 'sorted_weights', 'block_offsets']:
        old = getattr(b, name)
        setattr(b, name, guarded(old.numel(), old.dtype))
    tm = guarded(grid + 1, torch.int32)
    out = guarded(b.max_sorted * (H // 32), torch.uint8)

    def run():
        moe_sort_tile_map(b, ids, weights, tm, grid, n, E=E)
        scale_sort_tile_map(xs, b.sorted_ids, b.sorted_expert_ids, b.num_valid_ids,
                            out, tm, grid, n, bm, with_tile_map=False)

    run()
    torch.cuda.synchronize()
    counts = torch.bincount(ids.long().flatten(), minlength=E)
    padded = (counts + bm - 1) // bm * bm
    starts = padded.cumsum(0) - padded
    nv = int(padded.sum())
    assert b.num_valid_ids.tolist() == [nv, int(starts[-1])]
    assert torch.equal(b.sorted_expert_ids[:nv // bm].long(),
                       torch.repeat_interleave(torch.arange(E, device='cuda'), padded // bm))
    pair_cta = torch.arange(n * TOPK, device='cuda') // ((n * TOPK + SORT_CTAS - 1) // SORT_CTAS)
    want_hist = torch.bincount(pair_cta * E + ids.long().flatten(), minlength=SORT_CTAS * E)
    assert torch.equal(b.block_offsets.long(), want_hist)
    sid = b.sorted_ids[:nv].long()
    tok, slot = sid & 0xFFFFFF, sid >> 24
    valid = tok < n
    pair = tok[valid] * TOPK + slot[valid]
    assert torch.equal(pair.sort().values, torch.arange(n * TOPK, device='cuda'))
    row_e = torch.repeat_interleave(torch.arange(E, device='cuda'), padded)
    assert torch.equal(row_e[valid], ids.flatten().long()[pair])
    assert torch.equal(b.sorted_weights[:nv][valid], weights.flatten()[pair])
    assert torch.equal(b.sorted_weights[:nv][~valid], torch.zeros_like(b.sorted_weights[:nv][~valid]))
    within = torch.arange(nv, device='cuda') - torch.repeat_interleave(starts, padded)
    assert torch.equal(~valid, within >= torch.repeat_interleave(counts, padded))
    independent_tm = tile_map_ref(b.sorted_expert_ids, b.num_valid_ids, grid, bm)
    assert torch.equal(tm, independent_tm)
    gathered = torch.zeros((nv, H // 32), dtype=torch.uint8, device='cuda')
    gathered[valid] = xs[tok[valid]]
    want_scale = e8m0_shuffle(gathered).view(torch.uint8).flatten()[:nv * (H // 32)]
    assert torch.equal(out[:nv * (H // 32)], want_scale)
    for whole in guards:
        assert torch.all(whole[:64] == 37) and torch.all(whole[-64:] == 37)
    # Rebuild all persistent sort state, then replay over poisoned output buffers.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for name in ['sorted_ids', 'sorted_expert_ids', 'num_valid_ids', 'block_offsets']:
        getattr(b, name).fill_(-1)
    b.sorted_weights.fill_(float('nan'))
    tm.fill_(-1)
    out.fill_(165)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    assert b.num_valid_ids.tolist() == [nv, int(starts[-1])]
    assert torch.equal(tm, independent_tm)
    # Sorting may change row order; compare scales to this replay's row order.
    tok = b.sorted_ids[:nv].long() & 0xFFFFFF
    valid = tok < n
    gathered.zero_()
    gathered[valid] = xs[tok[valid]]
    want_scale = e8m0_shuffle(gathered).view(torch.uint8).flatten()[:nv * (H // 32)]
    assert torch.equal(out[:nv * (H // 32)], want_scale)
    for whole in guards:
        assert torch.all(whole[:64] == 37) and torch.all(whole[-64:] == 37)
