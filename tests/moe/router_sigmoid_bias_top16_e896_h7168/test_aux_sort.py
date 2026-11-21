# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The BM=16 MoE sort and the zeroed moe buffer the router launch emits: bit for bit against ``moe_sort_reference`` on the
launch's own top-16, and the same blocks (as sets of routes) as AITER's auxiliary sort."""
import pytest
import torch

from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168 import (make_aux, router_sigmoid_bias_top16_e896_h7168,
                                                                  shuffle_weight)
from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168.reference import E, H, TOPK, moe_sort_reference

MS = [1, 3, 12, 16, 17, 40, 65, 80, 128, 160, 192]


@pytest.fixture(scope="module")
def params():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return shuffle_weight(w), torch.randn(E, device="cuda", generator=g) * 0.1


def run(m, ws, bias, skew=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, H, device="cuda", generator=g).to(torch.bfloat16)
    if skew is not None:  # every token the same: all routes to the same 16 experts
        x = x[:1].repeat(m, 1)
    aux = make_aux(m, x.device)
    aux["moe_buf"].fill_(7.0)
    out = (torch.empty(m, TOPK, device="cuda"), torch.empty(m, TOPK, dtype=torch.int32, device="cuda"))
    router_sigmoid_bias_top16_e896_h7168(x, ws, bias, out=out, shuffled=True, aux=aux)
    return out, aux


def check(out, aux, m):
    weights, ids = out
    sti, sw, sei, nv, mi, rs = moe_sort_reference(ids, weights)
    total = int(nv[0])
    assert torch.equal(aux["num_valid_ids"], nv)
    for name, ref in (("sorted_token_ids", sti), ("sorted_weights", sw), ("m_indices", mi)):
        assert torch.equal(aux[name].view(torch.int32), ref.view(torch.int32)), name
    assert torch.equal(aux["sorted_expert_ids"][:total // 16], sei[:total // 16])
    assert torch.equal(aux["reverse_sorted"], rs)
    assert not aux["moe_buf"].view(torch.int16).any()


@pytest.mark.parametrize("m", MS)
def test_sort_matches_reference(params, m):
    ws, bias = params
    out, aux = run(m, ws, bias)
    check(out, aux, m)


def test_repeated_calls_and_hot_experts(params):
    ws, bias = params
    for seed in range(3):
        out, aux = run(80, ws, bias, seed=seed)
        check(out, aux, 80)
    out, aux = run(192, ws, bias, skew=True)  # 192 tokens on the same 16 experts: 12 blocks per expert
    check(out, aux, 192)


@pytest.mark.parametrize("m", [12, 80, 192])
def test_same_blocks_as_aiter(params, m):
    from aiter import fused_moe

    ws, bias = params
    out, aux = run(m, ws, bias)
    weights, ids = out
    ref = fused_moe.moe_sorting(ids, weights, E, 3584, torch.bfloat16, 16, None, None,
                                output_aux=fused_moe.AUX_SORT_OPUS, accumulate=True)
    r_sti, r_sw, r_sei, r_nv = ref[0], ref[1], ref[2], ref[3]
    total = int(r_nv[0])
    assert int(aux["num_valid_ids"][0]) == total and int(aux["num_valid_ids"][1]) == int(r_nv[1])
    assert torch.equal(aux["sorted_expert_ids"][:total // 16], r_sei[:total // 16])
    # AITER's two aux sorts pad with M or with M | topk << 24; the kernels read the token id masked to 24 bits
    def norm(sti):
        sti = sti.clone()
        sti[(sti & 0xFFFFFF) >= m] = m
        return sti

    def blocks(sti, sw):  # the routes of every 16-row block, order inside the block ignored
        key = (norm(sti[:total]).long() << 32) | (sw[:total].view(torch.int32).long() & 0xFFFFFFFF)
        return key.view(-1, 16).sort(1)[0]

    assert torch.equal(blocks(aux["sorted_token_ids"], aux["sorted_weights"]), blocks(r_sti, r_sw))


def test_back_to_back_sorts(params):
    """Calls of different M back to back with fresh aux tensors (the ids' parity tag): every call's sort equals the reference."""
    ws, bias = params
    for rep in range(3):
        for m in (12, 80, 17, 12, 192, 1, 80, 33, 192, 16):
            out, aux = run(m, ws, bias, seed=rep * 7 + m)
            check(out, aux, m)
