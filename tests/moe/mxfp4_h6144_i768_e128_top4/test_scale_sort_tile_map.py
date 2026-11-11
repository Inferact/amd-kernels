# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The merged activation-scale sort + gemm1 tile map (``common.scale_sort_tile_map``) after the BM256 host's routing sort
(``common.sort_count_all`` / ``common.sort_tile_map``, 256-row blocks) against the torch gather + AITER shuffle and the
torch tile map."""

import pytest
import torch
from aiter.utility.fp4_utils import e8m0_shuffle
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4 as prefill
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.scale_sort_tile_map import scale_sort_tile_map
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort import SortBuffers
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort_count_all import moe_sort_count_all
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.sort_tile_map import moe_sort_tile_map
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.tile_map import CLAIM_COUNTERS, tile_map_grid
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import E, H, TOPK, I

from .workload import make_routing, require_gfx950, tile_map_ref

SERVED_TOKENS = [513, 1024, 2048, 4096, 8192, 16384, 32768]

BM = prefill.BM
NB_N = I // 128


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def run_sort(n, ids, w):
    bufs = SortBuffers.allocate(n, E, TOPK, BM, "cuda")
    grid = tile_map_grid(bufs.max_sorted // BM, I)
    scratch = torch.empty((grid + 1,), dtype=torch.int32, device="cuda")
    if n <= prefill.SORT_COUNT_ALL_MAX_TOKENS:
        moe_sort_count_all(bufs, ids, w, scratch, grid, n, E=E, I=I)
    else:
        moe_sort_tile_map(bufs, ids, w, scratch, grid, n, E=E)
    return bufs


@pytest.mark.parametrize("n", SERVED_TOKENS)
def test_scale_sort_tile_map_matches_torch(n):
    w, ids = make_routing(n, seed=n + 7, skewed=True)
    g = torch.Generator(device="cuda").manual_seed(n)
    xs = torch.randint(100, 140, (n, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    bufs = run_sort(n, ids, w)
    rows = bufs.max_sorted
    nv = int(bufs.num_valid_ids[0])
    grid = tile_map_grid(rows // BM, I)
    gbj = 0

    def run():
        a_s = torch.full(((rows + 31) // 32 * 32, H // 32), 0xA5, dtype=torch.uint8, device="cuda")
        tm = torch.full((grid + 1,), 0x5A5A5A5A, dtype=torch.int32, device="cuda")
        scale_sort_tile_map(
            xs, bufs.sorted_ids, bufs.sorted_expert_ids, bufs.num_valid_ids, a_s, tm,
            grid, n, BM, a_group_bj=gbj,
        )  # fmt: skip
        return a_s, tm

    a_s, tm = run()
    # scales: each valid row its token's bytes, padding rows 0, AITER-shuffled;
    # tiles past num_valid are not written (the GEMM never reads them)
    sid = bufs.sorted_ids[:rows].long()
    tok = sid & 0xFFFFFF
    valid = (torch.arange(rows, device="cuda") < nv) & (tok < n)
    gathered = torch.zeros(((rows + 31) // 32 * 32, H // 32), dtype=torch.uint8, device="cuda")
    gathered[:rows][valid] = xs[tok[valid]]
    want = e8m0_shuffle(gathered).view(torch.uint8).reshape(-1)
    written = (nv + 31) // 32 * 32 * (H // 32)
    assert torch.equal(a_s.reshape(-1)[:written], want[:written])
    # tile map: the torch table, entry for entry
    assert torch.equal(tm, tile_map_ref(bufs.sorted_expert_ids, bufs.num_valid_ids, grid, BM))
    # and its contents: every valid (m-block, n-tile) once, -1 after, the count last
    valid_blocks = nv // BM
    entries = tm[: valid_blocks * NB_N].long()
    n_bits = (NB_N - 1).bit_length()
    m_blk, n_tile = entries >> n_bits, entries & ((1 << n_bits) - 1)
    key = torch.sort(m_blk * NB_N + n_tile).values
    assert torch.equal(key, torch.arange(valid_blocks * NB_N, device="cuda"))
    assert bool((tm[valid_blocks * NB_N : grid] == -1).all())
    assert grid - valid_blocks * NB_N >= CLAIM_COUNTERS
    assert int(tm[grid]) == valid_blocks * NB_N
    for _ in range(2):
        a2, tm2 = run()
        assert torch.equal(a2.reshape(-1)[:written], a_s.reshape(-1)[:written])
        assert torch.equal(tm2, tm)
