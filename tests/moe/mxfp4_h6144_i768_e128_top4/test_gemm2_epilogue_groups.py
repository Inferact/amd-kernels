# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Wave-private epilogue skips retain every valid row group and CTA barrier."""
import math

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_bm128 as aw
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from .test_entry import weights
from .workload import make_activation, make_routing, two_launch_slots


@pytest.mark.parametrize("m", [513, 640, 704, 705, 1024, 1536, 1537, 2048, 2049, 3072, 3584, 4096])
def test_changed_guarded_group_boundaries(weights, monkeypatch, m):
    """Against the two launches (full-height GEMM1, unconditional GEMM2 epilogue): counts cross every 16-row boundary,
    include repeated full tiles and change inside one graph."""
    _, w = weights
    wt, _ = make_routing(m, seed=m)

    def routing(changed):
        counts = [1] + [b + d for b in range(16, 129, 16) for d in (-1, 0, 1)]
        if changed:
            counts.reverse()
        counts.append(m * 4 - sum(counts))
        return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).int().cuda().view(m, 4)

    ids = routing(False)
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
        ids.copy_(routing(changed))
        q, s = mxfp4_quant(make_activation(m, seed=m + int(changed)))
        xq.copy_(q)
        xs.copy_(s)
        expect = expected()
        for _, value, _ in allocations:
            value.view(torch.uint8).fill_(0xFF)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            assert torch.equal(out, expect)
            assert all(int(t.count_nonzero()) == 0 for t in aw._ready.values())
        for storage, _, guard in allocations:
            assert (storage[:guard].view(torch.uint8) == 0xA5).all()
            assert (storage[-guard:].view(torch.uint8) == 0xA5).all()
