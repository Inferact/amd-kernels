# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Redundant experts (EPLB): 52 and 56 local experts against the reference, both entries, both routing distributions."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import (ExpertWeights, host, mxfp4_experts, mxfp4_experts_rows,
                                                        reference as ref)

from .workload import TOPK, make_activation, make_experts, make_routes, make_sorted_rows, require_gfx950

EXPERTS = (52, 56)


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


@pytest.fixture(scope="module", params=EXPERTS)
def experts(request):
    require_gfx950()
    raw = make_experts(seed=3, e=request.param)
    return request.param, raw, ExpertWeights.from_row_major(*raw)


@pytest.mark.parametrize("dist", ["uniform", "skew"])
@pytest.mark.parametrize("n", [16, 80, 192, 449])
@torch.inference_mode()
def test_tokens_match_reference_and_repeat(experts, n, dist):
    e, raw, w = experts
    assert w.n_experts == e
    xc, xs = make_activation(n, seed=n)
    ids, wt = make_routes(n, dist, seed=n, e=e)
    got = mxfp4_experts(xc, xs, w, wt, ids)
    expected = ref.experts_reference(xc, xs, *raw, wt, ids)
    assert cosine(got, expected) >= 0.99999
    assert torch.equal(got, mxfp4_experts(xc, xs, w, wt, ids))
    assert torch.equal(got, host.combine_slots(mxfp4_experts(xc, xs, w, wt, ids, return_slots=True), ids, ne=e))
    assert not host.ready_counters(xc.device).any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = mxfp4_experts(xc, xs, w, wt, ids)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(replayed, got)


@torch.inference_mode()
def test_ids_beyond_the_local_experts_route_nowhere(experts):
    e, raw, w = experts
    n = 33
    xc, xs = make_activation(n, seed=4)
    ids, wt = make_routes(n, "uniform", seed=4, e=e)
    ids[1::3, 2:] += e
    ids[0, 0] = -1
    got = mxfp4_experts(xc, xs, w, wt, ids)
    assert cosine(got, ref.experts_reference(xc, xs, *raw, wt, ids)) >= 0.99999


@pytest.mark.parametrize("dist", ["uniform", "skew"])
@pytest.mark.parametrize("n", [16, 192, 768])
@torch.inference_mode()
def test_rows_match_reference(experts, n, dist):
    e, raw, w = experts
    rows = n * TOPK
    offsets, weights = make_sorted_rows(rows, dist, seed=n, e=e)
    assert offsets.numel() == e + 1
    xc, xs = make_activation(rows, seed=n + 5)
    got = mxfp4_experts_rows(xc, xs, w, weights, offsets)
    assert cosine(got, ref.experts_rows_reference(xc, xs, *raw, weights, offsets)) >= 0.99999
    assert torch.equal(got, mxfp4_experts_rows(xc, xs, w, weights, offsets))
