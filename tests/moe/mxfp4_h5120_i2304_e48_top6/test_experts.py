# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``mxfp4_experts`` against ``reference.experts_reference`` at the served token counts, both routing distributions."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import ExpertWeights, host, mxfp4_experts, reference as ref

from .workload import TOKENS, make_activation, make_experts, make_routes, require_gfx950

COS_MIN = 0.99999
ROW_COS_MIN = 0.9999  # one flipped e4m3 code of a route's intermediate moves a row's cosine by up to 3e-5


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


@pytest.fixture(scope="module")
def experts():
    require_gfx950()
    raw = make_experts()
    return raw, ExpertWeights.from_row_major(*raw)


def check(experts, xc, xs, ids, wt):
    raw, w = experts
    got = mxfp4_experts(xc, xs, w, wt, ids)
    expected = ref.experts_reference(xc, xs, *raw, wt, ids)
    assert cosine(got, expected) >= COS_MIN
    row = torch.nn.functional.cosine_similarity(got.double(), expected.double(), dim=1)
    assert row[expected.double().norm(dim=1) > 0].min() >= ROW_COS_MIN
    assert torch.isfinite(got.float()).all()
    return got


@pytest.mark.parametrize("dist", ["uniform", "skew"])
@pytest.mark.parametrize("n", TOKENS)
@torch.inference_mode()
def test_matches_reference_and_repeats(experts, n, dist):
    _, w = experts
    xc, xs = make_activation(n, seed=n)
    ids, wt = make_routes(n, dist, seed=n)
    got = check(experts, xc, xs, ids, wt)
    assert torch.equal(got, mxfp4_experts(xc, xs, w, wt, ids))
    # the fused launch (GEMM1, GEMM2 and combine) equals the three launches bit for bit and leaves its counters at 0
    assert torch.equal(got, host.combine_slots(mxfp4_experts(xc, xs, w, wt, ids, return_slots=True), ids))
    assert not host.ready_counters(xc.device).any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = mxfp4_experts(xc, xs, w, wt, ids)
    for _ in range(3):
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, got)


@pytest.mark.parametrize("n", [1, 2, 7, 17, 33, 65, 97, 129, 161, 193, 250, 385, 449, 500, 1024])
@torch.inference_mode()
def test_every_dispatch_tier(experts, n):
    xc, xs = make_activation(n, seed=100 + n)
    ids, wt = make_routes(n, "uniform", seed=100 + n)
    check(experts, xc, xs, ids, wt)


@torch.inference_mode()
def test_slots_are_bf16_weighted_per_route(experts):
    raw, w = experts
    n = 40
    xc, xs = make_activation(n, seed=3)
    ids, wt = make_routes(n, "uniform", seed=3)
    slots = mxfp4_experts(xc, xs, w, wt, ids, return_slots=True)
    expected = ref.experts_reference(xc, xs, *raw, wt, ids, return_slots=True)
    assert slots.shape == (n, 6, 5120) and slots.dtype == torch.bfloat16
    assert cosine(slots, expected) >= COS_MIN
    assert torch.equal(host.combine_slots(slots, ids), ref._ordered_sum(slots))


@torch.inference_mode()
def test_swiglu_clamp_and_zero_rows(experts):
    """Activations ~40x larger push gate and up past the limit; zero rows (graph padding) give zero outputs."""
    n = 192
    xc, xs = make_activation(n, seed=9, scale=40.0)
    xc, xs = xc.clone(), xs.clone()
    xc.view(torch.uint8)[::5] = 0
    xs[::5] = 0
    ids, wt = make_routes(n, "uniform", seed=9)
    got = check(experts, xc, xs, ids, wt)
    assert got[::5].count_nonzero() == 0


@pytest.mark.parametrize("n", [16, 128, 384])
@torch.inference_mode()
def test_one_hot_expert_set(experts, n):
    """Every token on the same 6 experts: 6 experts with n rows, 42 idle."""
    xc, xs = make_activation(n, seed=n + 1)
    _, wt = make_routes(n, "uniform", seed=n + 1)
    ids = torch.arange(6, device="cuda", dtype=torch.int32).repeat(n, 1)
    check(experts, xc, xs, ids, wt)


@torch.inference_mode()
def test_layouts_match_aiter():
    require_gfx950()
    from aiter.ops.shuffle import shuffle_weight_a16w4

    g = torch.Generator(device="cuda").manual_seed(0)
    w13 = torch.randint(0, 256, (2, 4608, 2560), dtype=torch.uint8, device="cuda", generator=g)
    w2 = torch.randint(0, 256, (2, 5120, 1152), dtype=torch.uint8, device="cuda", generator=g)
    shuffled13 = shuffle_weight_a16w4(w13.clone(), 16, True).view(torch.uint8)
    shuffled2 = shuffle_weight_a16w4(w2.clone(), 16, False).view(torch.uint8)
    assert torch.equal(shuffled13.flatten(), ref.w13_layout(w13).flatten())
    assert torch.equal(shuffled2.flatten(), ref.w2_layout(w2).flatten())


@pytest.mark.parametrize("n", [16, 192])
@torch.inference_mode()
def test_ids_outside_the_local_experts_are_not_routed(experts, n):
    """Padding rows (id -1, as the router writes them) and experts of another rank (ids >= 48) route nowhere here."""
    xc, xs = make_activation(n, seed=n + 7)
    ids, wt = make_routes(n, "uniform", seed=n + 7)
    ids = ids.clone()
    ids[::4] = -1
    ids[1::4, 2:] = ids[1::4, 2:] + 48
    ids[2::4, 0] = 383
    wt = wt.clone()
    wt[::4] = 0
    got = check(experts, xc, xs, ids, wt)
    assert got[::4].count_nonzero() == 0
