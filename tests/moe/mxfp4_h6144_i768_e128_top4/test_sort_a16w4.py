# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sort_a16w4.moe_sort_a16w4`` (the 16-512 token sort of the small-row experts), both placement modes (``rank``: row
= expert start + the histogram atomic's return; else a second cursor atomic), eager and replayed: num_valid, the block
-> expert list and per expert the set of (token | slot << 24, weight bits) rows equal a CPU reference (experts
ascending, padded with token = M, weight 0); sets, since LDS atomic order places the rows inside an expert."""
import collections

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sort_a16w4 import moe_sort_a16w4
from .workload import TOPK, make_routing, require_gfx950

E, BM = 128, 16


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def _routing(m, kind):
    wt, ids = make_routing(m, seed=31 + m, skewed=kind == "skewed")
    if kind == "one":  # every pair on expert 77 (the slots keep their own rows)
        ids = torch.full_like(ids, 77)
    elif kind == "four":  # every token on experts 3, 64, 65, 127
        ids = torch.tensor([3, 64, 65, 127], dtype=torch.int32, device="cuda").expand(m, TOPK).contiguous()
    return wt, ids


def _sets(sorted_ids, sorted_w, eids, nv, bm=BM):
    n = int(nv[0])
    rows = sorted_ids[:n].cpu().tolist()
    w = sorted_w[:n].view(torch.int32).cpu().tolist()
    be = eids[:n // bm].cpu().tolist()
    per = collections.defaultdict(list)
    for b, e in enumerate(be):
        per[e] += list(zip(rows[b * bm:(b + 1) * bm], w[b * bm:(b + 1) * bm]))
    return n, be, {e: sorted(v) for e, v in per.items()}


def _reference(wt, ids, bm=BM):
    m = ids.shape[0]
    idl, wl = ids.cpu().tolist(), wt.view(torch.int32).cpu().tolist()
    per = collections.defaultdict(list)
    for t in range(m):
        for s in range(TOPK):
            per[idl[t][s]].append((t | s << 24, wl[t][s]))
    n, be, out = 0, [], {}
    for e in sorted(per):
        rows = per[e]
        pad = -len(rows) % bm
        rows = sorted(rows + [(m, 0)] * pad)
        out[e] = rows
        be += [e] * (len(rows) // bm)
        n += len(rows)
    return n, be, out


# every SORT_TOKEN_TIERS edge (64 / 128 / 192 / 256 / 384 / 448 / 512) and the next count
@pytest.mark.parametrize("m", [16, 17, 31, 32, 33, 64, 65, 97, 128, 129, 192, 193, 255, 256, 257, 384, 385, 448,
                               449, 512])
@pytest.mark.parametrize("kind", ["uniform", "skewed", "one", "four"])
@pytest.mark.parametrize("rank", [True, False])
@pytest.mark.parametrize("bm", [16, 32])
def test_sort_sets_and_graph(m, kind, rank, bm):
    wt, ids = _routing(m, kind)
    ref = _reference(wt, ids, bm)
    out = moe_sort_a16w4(ids, wt, E, bm, rank=rank)
    torch.cuda.synchronize()
    assert _sets(*out, bm=bm) == ref
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        got = moe_sort_a16w4(ids, wt, E, bm, rank=rank)
    for seed in (5, 6):
        wt2, ids2 = make_routing(m, seed=seed * 1000 + m, skewed=kind == "skewed")
        wt.copy_(wt2)
        ids.copy_(ids2)
        for t in got:
            t.fill_(-7)
        g.replay()
        torch.cuda.synchronize()
        assert _sets(*got, bm=bm) == _reference(wt, ids, bm)
