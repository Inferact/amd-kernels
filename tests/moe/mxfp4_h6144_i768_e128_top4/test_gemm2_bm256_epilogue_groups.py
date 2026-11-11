# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""BM256 group skips and wave probes retain every valid row across poisoned replays."""
import math

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4 as aw
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_bm128 as a128
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from .test_entry import weights
from .workload import make_activation, make_routing, two_launch_slots


def _group_ids(m, changed=False):
    # Every 16-row boundary of one 256-row tile (15/16/17 ... 255/256/257), then
    # second-tile boundaries; the last expert takes the rest (full tiles).
    counts = [1] + [b + d for b in range(16, 257, 16) for d in (-1, 0, 1)]
    counts += [271, 272, 273, 383, 384, 385, 511, 512, 513]
    if changed:
        counts.reverse()
    counts.append(m * 4 - sum(counts))
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).int().cuda().view(m, 4)


@pytest.mark.parametrize("m", [4097, 6144, 7424, 7425, 8192, 8193, 16384, 32768])
def test_bm256_changed_guarded_group_boundaries(weights, monkeypatch, m):
    """Against the two launches (full-height GEMM1, unconditional GEMM2 epilogue): counts cross every 16-row boundary
    and change inside one graph; above 7424 rows this also covers the sixteen-lane padding ballot and the wave-private
    LDS write / read order without an explicit write drain, at six and four column chunks."""
    _, w = weights
    wt, _ = make_routing(m, seed=m)
    ids = _group_ids(m)
    xq, xs = mxfp4_quant(make_activation(m, seed=m))

    def run():
        return aw.a4w4_experts((xq, xs), w.w13, w.w2, w.a4w4_scales, wt, ids)

    def expected():
        return two_launch_slots((xq, xs), w, wt, ids)

    expect = expected()
    assert torch.equal(run(), expect)
    torch.cuda.synchronize()
    original = torch.empty
    allocations = []

    def guarded(*size, **kw):
        shape = tuple(size[0]) if isinstance(size[0], (tuple, list)) else size
        guard = 128 // torch.tensor([], dtype=kw.get("dtype", torch.float32)).element_size()
        storage = original((math.prod(shape) + 2 * guard,), **kw)
        value = storage[guard:-guard].view(shape)
        allocations.append((storage, value, guard))
        return value

    with monkeypatch.context() as mp:
        mp.setattr(torch, "empty", guarded)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
    for storage, _, _ in allocations:
        storage.view(torch.uint8).fill_(0xA5)
    for changed in (False, True, False):
        ids.copy_(_group_ids(m, changed))
        q, s = mxfp4_quant(make_activation(m, seed=m + int(changed)))
        xq.copy_(q)
        xs.copy_(s)
        expect = expected()
        for _, value, _ in allocations:
            value.view(torch.uint8).fill_(0xFF)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            assert torch.equal(out, expect)
            assert all(int(t.count_nonzero()) == 0 for t in a128._ready.values())
        for storage, _, guard in allocations:
            assert (storage[:guard].view(torch.uint8) == 0xA5).all()
            assert (storage[-guard:].view(torch.uint8) == 0xA5).all()


def test_bm256_one_launch_serves_the_group_body_to_7424():
    """The one launch serves the group body at 4097-7424; above it the
    half-row body stages output byte offsets."""
    assert aw.GEMM12_EPILOGUE_GROUPS_MAX_TOKENS == 7424
    assert aw.gemm12_for(4097) and aw.gemm12_for(8192) and not aw.gemm12_for(8193)
    ns = aw.GEMM2_N_SPLIT
    assert aw._gemm12(128, 4, ns, True).kernel_name == (
        f"kernel_gemm12_mxfp4_h6144_i768_bm256_half_rows_epilogue_groups_ns{ns}")
    assert aw._gemm12(128, 4, ns).kernel_name == f"kernel_gemm12_mxfp4_h6144_i768_bm256_half_rows_ns{ns}"
    k2 = aw._gemm2(128, 4, ns)
    assert k2.kernel_name == "kernel_gemm2_mxfp4_h6144_i768_bm256_output_offsets"


def test_bm256_host_flag_by_tokens(weights, monkeypatch):
    """7424 tokens reach the group body, 7425 the output-offset body."""
    _, w = weights
    seen = []
    real = aw._gemm12

    def spy(*a):
        seen.append(a[3] if len(a) > 3 else False)
        return real(*a)

    monkeypatch.setattr(aw, "_gemm12", spy)
    for m in (7424, 7425):
        wt, ids = make_routing(m, seed=m)
        xq, xs = mxfp4_quant(make_activation(m, seed=m))
        aw.a4w4_experts((xq, xs), w.w13, w.w2, w.a4w4_scales, wt, ids)
    assert seen == [True, False]
