# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The full shared expert copy: the BF16 entry's TP shard read in place (view == shard copy, bit for bit, at every
tier), ``shared_expert_full`` against ``shared_mxfp8_ref``, and the routed-only prepared input."""
from dataclasses import replace

import pytest
import torch

from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import (
    MoeWeights, SharedWeights, mxfp4_moe, prepare_mxfp4_moe, shared_expert_full, unsupported_reason)
from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.chain import WARM_UP_TOKENS
from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.reference import shared_mxfp8_ref
from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.shared_dense import shared_dense_full
from .workload import H, I, make_activation, make_experts, make_full_shared, replay_check, require_gfx950


@pytest.fixture(scope="module")
def full_weights():
    """(full raw, {rank: (MoeWeights over the full copy, MoeWeights over the rank's shard copy)})."""
    require_gfx950()
    raw, shards = make_full_shared()
    routed, _ = make_experts()
    g = torch.Generator(device="cuda").manual_seed(5)
    gate = torch.randn(128, H, device="cuda", generator=g) / H ** .5
    bias = torch.randn(128, device="cuda", generator=g) * 1e-2
    base = MoeWeights.from_row_major(gate, bias, *routed)
    out = {}
    for r in range(2):
        full = SharedWeights.from_row_major(*raw, tp_rank=r)
        copy = SharedWeights.from_row_major(*shards[r])
        assert full.full and full.tp_rank == r and not copy.full
        out[r] = (replace(base, shared_weights=full), replace(base, shared_weights=copy))
    return raw, out


def _same(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("m", WARM_UP_TOKENS)
def test_view_equals_shard_copy(full_weights, m):
    """Every shared kernel tier of the BF16 entry (folded, 4-wave, combined persistent; both ranks): the view of the
    full copy gives the shard copy's partials and reduced output bit for bit."""
    _, ws = full_weights
    x = make_activation(m, seed=m)
    for r in range(2):
        full, copy = ws[r]
        got, want = mxfp4_moe(x, full, return_partials=True), mxfp4_moe(x, copy, return_partials=True)
        assert _same(got, want), (m, r)
        assert torch.equal(mxfp4_moe(x, full), mxfp4_moe(x, copy)), (m, r)


@pytest.mark.parametrize("m", [1, 96, 640, 2048, 4096])
def test_view_replay_and_prepared(full_weights, m):
    _, ws = full_weights
    full, copy = ws[1]
    x = make_activation(m, seed=m + 1)
    eager, replay = replay_check(lambda: list(mxfp4_moe(x, full, return_partials=True)))
    assert _same(eager, replay)
    assert _same(replay, mxfp4_moe(x, copy, return_partials=True))
    prepared = prepare_mxfp4_moe(x, full)
    assert _same(mxfp4_moe(prepared, full, return_partials=True), mxfp4_moe(prepared, copy, return_partials=True))


@pytest.mark.parametrize("n", [1, 31, 1024, 1281, 2048, 2561, 4097, 8192, 16384])
def test_shared_expert_full(full_weights, n):
    """Against ``shared_mxfp8_ref`` (cosine >= .999 per token), eager == replay, on a side stream the same bits."""
    raw, ws = full_weights
    w = ws[0][0]
    x = make_activation(n, seed=n)
    local = prepare_mxfp4_moe(x, w)
    sw = w.shared_weights
    eager, replay = replay_check(lambda: [shared_expert_full(local.shared_codes, local.shared_scales, sw)])
    assert torch.equal(eager[0], replay[0]) and bool(torch.isfinite(replay[0]).all())
    ref = shared_mxfp8_ref(x[:256], *raw)
    cos = torch.nn.functional.cosine_similarity(replay[0][:256].float(), ref, dim=1)
    assert cos.min() >= .999, (n, cos.min().item())
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    out = shared_expert_full(local.shared_codes, local.shared_scales, sw, stream=side)
    torch.cuda.current_stream().wait_stream(side)
    assert torch.equal(out, eager[0])


@pytest.mark.parametrize("n", [1024, 1280, 2048, 4096])
def test_full_schedules_bitwise(full_weights, n):
    """Every gate/up schedule and tile of the full expert stores the same bits (the sweep's candidates)."""
    _, ws = full_weights
    w = ws[0][0]
    local = prepare_mxfp4_moe(make_activation(n, seed=3), w)
    q, s, sw = local.shared_codes, local.shared_scales, w.shared_weights
    want = shared_dense_full(q, s, sw, config=(((4, 4), "deferred"), ((4, 4), False)))
    for t1, sched in (((2, 4), "deferred"), ((4, 4), "early"), ((4, 4), "single"), ((2, 4), "single")):
        if sched == "single" and ((n + 255) // 256) * 24 * (4 // t1[0]) > 256:
            continue
        for t2 in ((4, 4), (2, 4)):
            got = shared_dense_full(q, s, sw, config=((t1, sched), (t2, True)))
            assert torch.equal(got, want), (n, t1, sched, t2)


@pytest.mark.parametrize("m", [1, 17, 96, 513, 640, 641, 2048, 3584, 4096, 4097, 8192, 32768])
def test_routed_only_partials(full_weights, m):
    """The routed-only prepared input: the BF16 entry's routed slots bit for bit, no shared output."""
    _, ws = full_weights
    w = ws[0][0]
    x = make_activation(m, seed=m)
    routed = prepare_mxfp4_moe(x, w).routed()
    assert routed.routed_only and routed.shared_codes is None
    eager, replay = replay_check(lambda: [mxfp4_moe(routed, w, return_partials=True)[0]])
    slots, shared = mxfp4_moe(routed, w, return_partials=True)
    assert shared is None and torch.equal(eager[0], replay[0]) and torch.equal(slots, eager[0])
    assert torch.equal(slots, mxfp4_moe(x, w, return_partials=True)[0]), m
    with pytest.raises(ValueError):
        mxfp4_moe(routed, w)


def test_unsupported_reason_full_and_shard():
    E = 128
    routed = (torch.empty((E, 2 * I, H // 2), dtype=torch.uint8), torch.empty((E, 2 * I, H // 32), dtype=torch.uint8),
              torch.empty((E, H, I // 2), dtype=torch.uint8), torch.empty((E, H, I // 32), dtype=torch.uint8))
    f8 = torch.float8_e4m3fn
    for i in (I, 2 * I):
        shared = (torch.empty((2 * i, H), dtype=f8), torch.empty((2 * i, H // 32), dtype=torch.uint8),
                  torch.empty((H, i), dtype=f8), torch.empty((H, i // 32), dtype=torch.uint8))
        assert unsupported_reason(*routed, *shared) is None, i
    bad = (torch.empty((4 * I, H), dtype=f8), torch.empty((4 * I, H // 32), dtype=torch.uint8),
           torch.empty((H, I), dtype=f8), torch.empty((H, I // 32), dtype=torch.uint8))
    assert unsupported_reason(*routed, *bad) is not None


def test_full_needs_rank():
    raw = (torch.empty((4 * I, H), dtype=torch.uint8), torch.empty((4 * I, H // 32), dtype=torch.uint8),
           torch.empty((H, 2 * I), dtype=torch.uint8), torch.empty((H, 2 * I // 32), dtype=torch.uint8))
    with pytest.raises(AssertionError):
        SharedWeights.from_row_major(*raw)
