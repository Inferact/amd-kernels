# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The BM256 and BM128 short bodies own every valid byte across the half-tile boundary."""
import math

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4 as aw
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4_bm128 as a128
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from .test_entry import weights
from .workload import make_activation, make_routing, two_launch_slots


def _boundary_ids(m, changed=False):
    counts = [1, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 383, 384, 385]
    if changed:
        counts = counts[::-1]
    counts.append(m * 4 - sum(counts))
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).int().cuda().view(m, 4)


@pytest.mark.parametrize("m", [4097, 4098, 5000, 7424, 7425, 8191, 8192, 8193, 16384, 24576, 32767, 32768, 65536])
def test_short_and_full_tiles_changed_poisoned_replay(weights, monkeypatch, m):
    """Against the two launches (full-height GEMM1): changed routing crosses 127/128/129 and 255/256/257 rows inside
    one graph, a long last expert forces full tiles, repeated claims and XCD tails; the six-chunk body stages output
    byte offsets in LDS (above the signed-i32 range at 65536 tokens), the four-chunk body packed ids."""
    _, w = weights
    wt, _ = make_routing(m, seed=m)
    ids = _boundary_ids(m)
    xq, xs = mxfp4_quant(make_activation(m, seed=m))
    # the one-launch range pinned: the test does not follow the host's measured bounds
    monkeypatch.setattr(aw, "GEMM12_TOKENS", (4097, 8192))

    def run():
        return aw.a4w4_experts((xq, xs), w.w13, w.w2, w.a4w4_scales, wt, ids)

    def expected():
        return two_launch_slots((xq, xs), w, wt, ids)

    expect = expected()
    assert torch.equal(run(), expect)
    torch.cuda.synchronize()
    original = torch.empty
    allocations = []

    def guarded_empty(*size, **kw):
        shape = tuple(size[0]) if isinstance(size[0], (tuple, list)) else size
        dtype = kw.get("dtype", torch.float32)
        guard = 128 // torch.tensor([], dtype=dtype).element_size()
        storage = original((math.prod(shape) + 2 * guard,), **kw)
        result = storage[guard:-guard].view(shape)
        allocations.append((storage, result, guard))
        return result

    with monkeypatch.context() as mp:
        mp.setattr(torch, "empty", guarded_empty)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
    for storage, _, _ in allocations:
        storage.view(torch.uint8).fill_(0xA5)
    for changed in (False, True, False):
        ids.copy_(_boundary_ids(m, changed))
        q, s = mxfp4_quant(make_activation(m, seed=m + int(changed)))
        xq.copy_(q)
        xs.copy_(s)
        expect = expected()
        for _, value, _ in allocations:
            value.view(torch.uint8).fill_(0xFF)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert torch.equal(out, expect)
        assert all(int(t.count_nonzero()) == 0 for t in a128._ready.values())
        for storage, _, guard in allocations:
            assert (storage[:guard].view(torch.uint8) == 0xA5).all()
            assert (storage[-guard:].view(torch.uint8) == 0xA5).all()


def _boundary_ids128(m, changed=False):
    counts = [1, 31, 32, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257]
    if changed:
        counts = counts[::-1]
    counts.append(m * 4 - sum(counts))
    return torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts)).int().cuda().view(m, 4)


@pytest.mark.parametrize("m", [513, 1024, 1407, 1408, 1536, 2048, 2049, 3072, 4095, 4096])
def test_bm128_short_and_full_tiles_changed_poisoned_replay(weights, monkeypatch, m):
    """The 128-row path's 64-row GEMM1 body against the two launches (full-height GEMM1): changed routing crosses
    63/64/65 and 127/128/129 rows inside one graph, every scratch guarded and poisoned."""
    _, w = weights
    wt, _ = make_routing(m, seed=m)
    ids = _boundary_ids128(m)
    xq, xs = mxfp4_quant(make_activation(m, seed=m))
    def run():
        return a128.a4w4_experts((xq, xs), w.w13, w.w2, w.a4w4_scales, wt, ids)

    def expected():
        return two_launch_slots((xq, xs), w, wt, ids)

    expect = expected()
    assert torch.equal(run(), expect)
    torch.cuda.synchronize()
    original = torch.empty
    allocations = []

    def guarded_empty(*size, **kw):
        shape = tuple(size[0]) if isinstance(size[0], (tuple, list)) else size
        dtype = kw.get("dtype", torch.float32)
        guard = 128 // torch.tensor([], dtype=dtype).element_size()
        storage = original((math.prod(shape) + 2 * guard,), **kw)
        result = storage[guard:-guard].view(shape)
        allocations.append((storage, result, guard))
        return result

    with monkeypatch.context() as mp:
        mp.setattr(torch, "empty", guarded_empty)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
    for storage, _, _ in allocations:
        storage.view(torch.uint8).fill_(0xA5)
    for changed in (False, True, False):
        ids.copy_(_boundary_ids128(m, changed))
        q, s = mxfp4_quant(make_activation(m, seed=m + int(changed)))
        xq.copy_(q)
        xs.copy_(s)
        expect = expected()
        for _, value, _ in allocations:
            value.view(torch.uint8).fill_(0xFF)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert torch.equal(out, expect)
        assert all(int(t.count_nonzero()) == 0 for t in a128._ready.values())
        for storage, _, guard in allocations:
            assert (storage[:guard].view(torch.uint8) == 0xA5).all()
            assert (storage[-guard:].view(torch.uint8) == 0xA5).all()
