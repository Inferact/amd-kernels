# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry at every M: MXFP4 x MXFP4 (one activation quant rule), per-slot stores, the shared expert's slots."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, mxfp4_moe
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import mxfp4_moe_experts
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import _prepare, _route_quant_small, route, variant_for
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import a16w4_moe_ref, w4a4_moe_ref

from .workload import (H, make_activation, make_experts, make_routing, make_shared, replay_check, require_gfx950,
                       small_slots, two_launch_slots)


@pytest.fixture(scope="module")
def weights():
    require_gfx950()
    raw, _ = make_experts()
    gen = torch.Generator(device="cuda").manual_seed(76)
    gate = torch.randn((128, H), generator=gen, device="cuda") / H**0.5
    bias = torch.randn((128,), generator=gen, device="cuda") * 0.01
    return raw, MoeWeights.from_row_major(gate, bias, *raw, shared=make_shared()[1])


def _routing(m, routing):
    wt, ids = make_routing(m, seed=m, skewed=routing == "hot")
    if routing == "four":
        ids = torch.arange(4, device="cuda", dtype=torch.int32).expand(m, 4).contiguous()
    if routing == "one-hot":
        # A single expert receives every token; the remaining slots stay distinct.
        ids = ids % 127 + 1
        ids[:, 0] = 0
    return wt, ids


def _w4a4_ref(x, raw, wt, ids):
    q, s = mxfp4_quant(x)
    return w4a4_moe_ref((q, s.view(torch.uint8)), *raw, wt, ids)


@pytest.mark.parametrize("m", [1, 2, 5, 8, 15, 16, 17, 32, 63, 128, 129, 191, 192, 207, 255, 256, 257, 300, 384, 385, 448, 511, 512])
@pytest.mark.parametrize("routing", ["uniform", "hot", "four", "one-hot"])
def test_small_m_w4a4_contract(weights, m, routing):
    """Below 513 tokens the routed slots (shared tiles folded in) against the FP32 reference on the same MXFP4
    operands, deterministic and finite."""
    raw, w = weights
    x = make_activation(m, seed=m)
    wt, ids = _routing(m, routing)
    parts, _ = small_slots(x, w, wt, ids)
    expect = _w4a4_ref(x, raw, wt, ids)
    assert torch.isfinite(parts).all()
    for _ in range(2):
        assert torch.equal(parts, small_slots(x, w, wt, ids)[0])
    got = sum(parts[:, k].float() for k in range(4))
    cos = torch.nn.functional.cosine_similarity(got, expect, dim=1)
    assert cos.min() >= 0.999, cos.min().item()


@pytest.mark.parametrize("m", [1, 8, 32, 128, 256, 512])
def test_small_m_against_bf16_activations(weights, m):
    """Against BF16 activations the entry is as close as the ideal W4A4 emulation of its quant rule (minus 0.001)."""
    raw, w = weights
    x = make_activation(m, seed=m)
    wt, ids = _routing(m, "uniform")
    hi = a16w4_moe_ref(x, *raw, wt, ids).float()
    parts, _ = small_slots(x, w, wt, ids)
    got = sum(parts[:, k].float() for k in range(4))
    ideal = _w4a4_ref(x, raw, wt, ids)
    cos = torch.nn.functional.cosine_similarity(got.flatten(), hi.flatten(), dim=0)
    cos_ideal = torch.nn.functional.cosine_similarity(ideal.flatten(), hi.flatten(), dim=0)
    assert cos >= cos_ideal - 0.001, (cos.item(), cos_ideal.item())


def test_one_quant_rule_at_every_m(weights):
    """The entry's MXFP4 codes and scales of the same rows are bit-equal whichever M (and so router / quant kernel)
    they arrive in; a token's routed slots agree across M to cos >= 0.9999."""
    _, w = weights
    full = make_activation(4097, seed=4097)
    q_ref, s_ref = mxfp4_quant(full)
    parts = {}
    for m in [1, 8, 64, 65, 512, 513, 1025, 2048, 2049, 4097]:
        x = full[:m].contiguous()
        q, s = _prepare(x, w)[0] if m >= 1025 else _route_quant_small(x, w)[0]
        assert torch.equal(q, q_ref[:m]), m
        assert torch.equal(s.view(torch.uint8), s_ref[:m].view(torch.uint8)), m
        parts[m] = mxfp4_moe(x, w, return_partials=True)[0][:1].float()
    for m, p in parts.items():
        cos = torch.nn.functional.cosine_similarity(p.flatten(), parts[1].flatten(), dim=0)
        assert cos >= 0.9999, (m, cos.item())


@pytest.mark.parametrize("m", [8, 16, 17, 128, 256, 257, 385, 448, 512, 1024, 2048, 4096, 8192, 16384, 32768])
def test_entry_routing_and_replay(weights, m):
    _, w = weights
    x = make_activation(m, seed=m, scale=1.0)
    wt, ids = route(x, w)
    scores = (x.double() @ w.router_weight.double().t()).sigmoid()
    expect_ids = (scores + w.correction_bias.double()).topk(4, dim=1).indices
    assert torch.equal(ids, expect_ids.int())
    expect_wt = scores.gather(1, expect_ids)
    expect_wt = expect_wt / expect_wt.sum(1, keepdim=True) * 2
    torch.testing.assert_close(wt.double(), expect_wt, atol=1e-3, rtol=0)
    for partial in [False, True]:
        eager, replayed = replay_check(lambda: list(mxfp4_moe(x, w, return_partials=True)) if partial
                                       else [mxfp4_moe(x, w)])
        assert all(torch.equal(a, b) for a, b in zip(eager, replayed))


def test_dispatch():
    """W4A4 at every M: no a16w4 / a8w4 range is served."""
    assert [variant_for(m) for m in [1, 8, 191, 192, 257, 384, 512, 513, 1024, 2048, 4096, 4097, 32768]] == [
        "a4w4_small"] * 7 + ["a4w4_bm128"] * 2 + ["a4w4_bm128_router_quant"] * 2 + ["a4w4_router_quant"] * 2
    for m in [0, -1, 87382]:
        with pytest.raises(ValueError):
            variant_for(m)



@pytest.mark.parametrize("m", [513, 1024, 2049, 4096, 8192])
def test_a4w4_entry_partials_equal_two_launch(weights, m):
    """The BM128 and one-launch BM256 hosts' slots equal the BM256 host's two launches bit for bit."""
    _, w = weights
    x = make_activation(m, seed=m)
    wt, ids = make_routing(m, seed=m, skewed=True)
    xq = mxfp4_quant(x)
    assert torch.equal(two_launch_slots(xq, w, wt, ids), mxfp4_moe_experts(xq, wt, ids, w))


@pytest.mark.parametrize("m", [513, 4096, 8193, 16384])
def test_hosts_against_reference(weights, m):
    """The BM128 and BM256 hosts' slots (the oracle of the one-launch and short-tile tests) against the FP32 reference
    on the same MXFP4 operands: relative L2 of the slot sum < 4e-3 (README contract)."""
    raw, w = weights
    x = make_activation(m, seed=m)
    wt, ids = make_routing(m, seed=m, skewed=True)
    parts = mxfp4_moe_experts(mxfp4_quant(x), wt, ids, w)
    got = sum(parts[:, k].float() for k in range(4))
    ref = _w4a4_ref(x, raw, wt, ids)
    assert ((got - ref).norm() / ref.norm()).item() < 4e-3


@pytest.mark.parametrize("m", [1, 3, 5, 8, 12, 15, 17, 191, 192, 257, 384, 385, 448, 480, 512, 513, 1024, 1536, 1537, 1792, 2048, 2560, 3072, 3583, 3584, 4096, 4097])
def test_output_and_scratch_guards(weights, monkeypatch, m):
    """Every Python-allocated scratch / output has 128-byte guards on both sides; NaN-poisoned outputs also catch the
    buffer-resource OOB stores the hardware drops instead of writing a guard."""
    import math
    _, w = weights
    x = make_activation(m, seed=m)
    original = torch.empty
    guarded = []

    def allocate(*size, **kwargs):
        dims = tuple(size[0]) if isinstance(size[0], (tuple, list)) else size
        dtype = kwargs.get("dtype", torch.float32)
        guard = 128 // torch.tensor([], dtype=dtype).element_size()
        storage = original((math.prod(dims) + 2 * guard,), **kwargs)
        storage.fill_(17)
        result = storage[guard:-guard].view(dims)
        if dtype in (torch.bfloat16, torch.float32):
            result.fill_(float("nan"))
        guarded.append((storage, guard))
        return result

    monkeypatch.setattr(torch, "empty", allocate)
    for partial in (False, True):
        out = mxfp4_moe(x, w, return_partials=True) if partial else (mxfp4_moe(x, w),)
        assert all(torch.isfinite(t).all() for t in out)
    torch.cuda.synchronize()
    assert guarded
    for storage, guard in guarded:
        assert (storage[:guard] == 17).all()
        assert (storage[-guard:] == 17).all()


@pytest.mark.parametrize("m", [1, 8, 192, 257, 384, 512])
def test_all_slots_on_one_expert(weights, m):
    raw, w = weights
    x = make_activation(m, seed=m)
    ids = torch.zeros((m, 4), device="cuda", dtype=torch.int32)
    wt = torch.full((m, 4), 0.5, device="cuda")
    parts, _ = small_slots(x, w, wt, ids)
    ref = _w4a4_ref(x, raw, wt, ids)
    cosine = torch.nn.functional.cosine_similarity(sum(parts[:, k].float() for k in range(4)), ref, dim=1)
    assert cosine.min() >= 0.999
    assert torch.equal(parts, small_slots(x, w, wt, ids)[0])
