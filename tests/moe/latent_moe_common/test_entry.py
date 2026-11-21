# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The router -> experts chain through the entries is bit for bit the direct shape-package chain (router + sort aux,
activation quantizer, experts summed and as the slot buffer, AITER sort) at the smallest and the largest served rows,
and every reason is None on those inputs."""
import importlib

import pytest
import torch

from amd_kernels.moe import latent_moe_common as E
from amd_kernels.moe.latent_moe_common import experts, router
from tests.moe.mxfp4_h3584_i3072_e896_top16.workload import (
    make_checkpoint,
    make_routing,
)

pytestmark = pytest.mark.skipif(not E.available(), reason="needs gfx950 with FlyDSL")
GATE = (896, 7168, 16)
W13 = (896, 768, 1792)


@pytest.fixture(scope="module")
def chain():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(896, 7168, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn(896, device="cuda", generator=g) * 0.1
    rh = importlib.import_module(router._SHAPE_HOSTS[GATE])
    eh = importlib.import_module(experts._SHAPE_HOSTS[W13])
    ck = eh.MoeWeights.from_checkpoint(*make_checkpoint())
    return w, bias, rh, eh, E.moe_weights(ck.w13, ck.s13, ck.w2, ck.s2)


def same(a, b):
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("m", [1, 192])
def test_chain_is_the_package_chain(chain, m):
    w, bias, rh, eh, weights = chain
    dev = w.device
    x = (torch.randn(m, 7168, device=dev) * 0.5).to(torch.bfloat16)
    lat = (torch.randn(m, 3584, device=dev) * 0.5).to(torch.bfloat16)

    aux_d = rh.make_aux(m, dev, False)
    tw_d, ti_d = rh.router_sigmoid_bias_top16_e896_h7168(x, w, bias, 1.0, shuffled=False, aux=aux_d)
    sort_d = eh.Sort.from_aux(aux_d)
    codes_d, scales_d = eh.quantize_activations(lat)
    out_d = eh.mxfp4_moe_experts(codes_d, scales_d, weights, sort_d)
    slots_d = eh.mxfp4_moe_experts(codes_d, scales_d, weights, sort_d, reduce=False).clone()
    torch.cuda.synchronize()

    assert E.router_sigmoid_bias_topk_weight_reason(w, bias) is None
    aux = E.router_aux(w, m, dev, zero_moe_buf=False)
    assert E.router_sigmoid_bias_topk_unsupported_reason(x, w, bias, 1.0, aux=aux) is None
    tw, ti = E.router_sigmoid_bias_topk(x, w, bias, 1.0, aux=aux)
    assert E.moe_sort_from_aux_unsupported_reason(aux, weights=weights) is None
    sort = E.moe_sort_from_aux(aux, weights=weights)
    assert E.quantize_moe_activations_unsupported_reason(lat, weights=weights) is None
    codes, scales = E.quantize_moe_activations(lat, weights=weights)
    assert E.mxfp4_moe_experts_unsupported_reason(codes, scales, weights, sort) is None
    out = E.mxfp4_moe_experts(codes, scales, weights, sort)
    slots = E.mxfp4_moe_experts(codes, scales, weights, sort, reduce=False)
    torch.cuda.synchronize()

    assert same(tw, tw_d) and same(ti, ti_d)
    nv = int(aux["num_valid_ids"][0])
    valid = {"sorted_expert_ids": nv // 16}  # the blocks past the valid rows are not written
    for k, t in aux.items():
        if t is not None:
            n = valid.get(k, t.numel())
            assert same(t[:n], aux_d[k][:n]), k
    assert same(codes, codes_d) and same(scales, scales_d)
    assert same(out, out_d)
    rows = aux["reverse_sorted"].long()
    assert same(slots[rows], slots_d[rows])
    assert E.moe_sort_capacity(m, weights=weights) == eh.sort_capacity(m)


@pytest.mark.parametrize("m", [1, 192])
def test_aiter_sort_is_the_package_sort(chain, m):
    _, _, _, eh, weights = chain
    ids, tw = make_routing(m)
    a, b = E.moe_sort(ids, tw, weights=weights), eh.moe_sort(ids, tw)
    assert same(a.num_valid_ids, b.num_valid_ids) and same(a.reverse_sorted, b.reverse_sorted)
    nv = int(a.num_valid_ids[0])
    for f, n in (("sorted_token_ids", nv), ("sorted_weights", nv), ("sorted_expert_ids", nv // 16)):
        assert same(getattr(a, f)[:n], getattr(b, f)[:n]), f


def test_router_without_sort_and_into_out(chain):
    w, bias, rh, _, _ = chain
    x = torch.randn(40, 7168, device=w.device).to(torch.bfloat16)
    out = (torch.empty(40, 16, device=w.device), torch.empty(40, 16, dtype=torch.int32, device=w.device))
    assert E.router_sigmoid_bias_topk_unsupported_reason(x, w, bias, 2.5, out) is None
    got = E.router_sigmoid_bias_topk(x, w, bias, 2.5, out)
    want = rh.router_sigmoid_bias_top16_e896_h7168(x, w, bias, 2.5)
    assert got[0] is out[0] and same(got[0], want[0]) and same(got[1], want[1])
