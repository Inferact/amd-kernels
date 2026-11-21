# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``mxfp4_experts_rows`` (rows already sorted by expert) and the bf16 input of both entries."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import (ExpertWeights, host, mxfp4_experts, mxfp4_experts_rows,
                                                        reference as ref)

from .workload import E, H, TOPK, make_activation, make_experts, make_routes, make_sorted_rows, require_gfx950


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


@pytest.fixture(scope="module")
def experts():
    require_gfx950()
    raw = make_experts(seed=2)
    return raw, ExpertWeights.from_row_major(*raw)


@pytest.mark.parametrize("dist", ["uniform", "skew"])
@pytest.mark.parametrize("n", [16, 64, 192, 768])
@torch.inference_mode()
def test_rows_match_reference_and_repeat(experts, n, dist):
    raw, w = experts
    rows = n * TOPK
    offsets, weights = make_sorted_rows(rows, dist, seed=n)
    xc, xs = make_activation(rows, seed=n + 5)
    got = mxfp4_experts_rows(xc, xs, w, weights, offsets)
    expected = ref.experts_rows_reference(xc, xs, *raw, weights, offsets)
    assert cosine(got, expected) >= 0.99999
    assert torch.equal(got, mxfp4_experts_rows(xc, xs, w, weights, offsets))
    assert not host.ready_counters(xc.device).any()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = mxfp4_experts_rows(xc, xs, w, weights, offsets)
    for _ in range(3):
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(replayed, got)


@torch.inference_mode()
def test_rows_are_the_slots_of_the_token_entry(experts):
    """The rows of a token's routes, sorted by expert, give the token entry's slots bit for bit."""
    _, w = experts
    n = 96
    xc, xs = make_activation(n, seed=1)
    ids, wt = make_routes(n, "skew", seed=1)
    slots = mxfp4_experts(xc, xs, w, wt, ids, return_slots=True)
    order = torch.argsort(ids.flatten().long(), stable=True)
    token = order // TOPK
    counts = torch.bincount(ids.flatten().long(), minlength=E)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), counts.cumsum(0)]).int()
    got = mxfp4_experts_rows(xc[token], xs[token], w, wt.flatten()[order], offsets)
    assert torch.equal(got, slots.view(-1, H)[order])


@torch.inference_mode()
def test_rows_capacity_beyond_the_row_count_and_empty_experts(experts):
    """A buffer larger than the received rows, 47 experts empty: only the rows below ``offsets[48]`` are written."""
    raw, w = experts
    valid, cap = 100, 150
    offsets = torch.zeros(E + 1, dtype=torch.int32, device="cuda")
    offsets[8:] = valid
    xc, xs = make_activation(cap, seed=4)
    weights = torch.rand(cap, device="cuda") + 0.1
    got = mxfp4_experts_rows(xc, xs, w, weights, offsets)
    expected = ref.experts_rows_reference(xc, xs, *raw, weights, offsets)
    assert cosine(got[:valid], expected[:valid]) >= 0.99999


@pytest.mark.parametrize("n", [1, 16, 100, 384])
@torch.inference_mode()
def test_bf16_input_is_quantized_in_the_routing_launch(experts, n):
    """The kernel's MXFP8 quant is ``reference.quantize_mxfp8`` bit for bit: bf16 rows and their codes agree."""
    _, w = experts
    x = torch.randn((n, H), device="cuda", generator=torch.Generator(device="cuda").manual_seed(n)).bfloat16()
    x[0] = 0
    xc, xs = ref.quantize_mxfp8(x)
    ids, wt = make_routes(n, "uniform", seed=n)
    assert torch.equal(mxfp4_experts(x, None, w, wt, ids), mxfp4_experts(xc, xs, w, wt, ids))
    offsets, weights = make_sorted_rows(6 * n, "uniform", seed=n)
    xr = torch.randn((6 * n, H), device="cuda").bfloat16()
    rc, rs = ref.quantize_mxfp8(xr)
    assert torch.equal(mxfp4_experts_rows(xr, None, w, weights, offsets),
                       mxfp4_experts_rows(rc, rs, w, weights, offsets))
