# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""router_sigmoid_bias_top16_e896_h7168 against its reference and AITER: the GEMM half on the fp64 logits, the top-16
half bit for bit against aiter.biased_grouped_topk on the logits it consumes (ties and slot order included), scaling,
both weight layouts, every tile configuration and K split."""
import pytest
import torch

from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168 import (host, router_logits,
                                                                  router_sigmoid_bias_top16_e896_h7168, shuffle_weight,
                                                                  top16, unshuffle_weight)
from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168.reference import (E, H, TOPK, logits_reference,
                                                                            top16_reference)

MS = [1, 2, 5, 12, 15, 16, 17, 24, 33, 40, 48, 65, 80, 96, 112, 128, 129, 160, 176, 192, 193, 300]
CONFIGS = [(ks, 4, 1) for ks in (4, 7, 8, 14, 16)] + [(ks, 4, 2) for ks in (14, 28)] + [(ks, 1, 8) for ks in (14, 28)] + [(14, 2, 4), (8, 4, 1, 2), (4, 4, 1, 3), (8, 4, 1, 3), (7, 4, 1, 2), (14, 4, 1, 2), (8, 1, 4, 2), (8, 2, 2, 2)]


def aiter_top16(logits, bias, scale=1.0):
    from aiter import biased_grouped_topk

    m = logits.shape[0]
    weights = torch.empty(m, TOPK, dtype=torch.float32, device=logits.device)
    ids = torch.empty(m, TOPK, dtype=torch.int32, device=logits.device)
    biased_grouped_topk(logits, bias, weights, ids, 1, 1, True, scale)
    return weights, ids


@pytest.fixture(scope="module")
def params():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn(E, device="cuda", generator=g) * 0.1
    return w, shuffle_weight(w), bias


def same_bits(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def check_logits(lg, ref):
    cos = torch.nn.functional.cosine_similarity(lg.double(), ref.double(), dim=1)
    assert bool(torch.all(cos >= 0.99999)), cos.min().item()
    assert (lg.double() - ref.double()).abs().max().item() < 1e-4


def check_top16(lg, bias, scale=1.0):
    """The kernel's top-16 of ``lg`` equals AITER's bit for bit, and the torch reference's within 2 ulp."""
    weights, ids = top16(lg, bias, scale)
    aw, aids = aiter_top16(lg, bias, scale)
    assert torch.equal(ids, aids) and same_bits(weights, aw)
    rw, rids = top16_reference(lg, bias, scale)
    assert torch.equal(ids, rids)
    assert torch.allclose(weights, rw, rtol=4 * 2.0**-23, atol=0)
    return ids


def test_shuffle_is_a_permutation(params):
    w, ws, _ = params
    assert torch.equal(unshuffle_weight(ws).view(torch.int16), w.view(torch.int16))


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("m", MS)
def test_router(params, shuffled, m):
    w, ws, bias = params
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    weights, ids = router_sigmoid_bias_top16_e896_h7168(x, ws if shuffled else w, bias, shuffled=shuffled)
    lg = router_logits(x, ws if shuffled else w, shuffled=shuffled)
    check_logits(lg, logits_reference(x, w))
    aw, aids = aiter_top16(lg, bias)
    assert weights.shape == (m, TOPK) and ids.dtype == torch.int32
    assert torch.equal(ids, aids) and same_bits(weights, aw)
    w2, i2 = router_sigmoid_bias_top16_e896_h7168(x, ws if shuffled else w, bias, shuffled=shuffled)
    assert same_bits(w2, weights) and torch.equal(i2, ids)


@pytest.mark.parametrize("m", [3, 40, 192])
def test_scale(params, m):
    w, ws, bias = params
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    weights, ids = router_sigmoid_bias_top16_e896_h7168(x, ws, bias, 2.5, shuffled=True)
    aw, aids = aiter_top16(router_logits(x, ws, shuffled=True), bias, 2.5)
    assert torch.equal(ids, aids) and same_bits(weights, aw)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("m", [3, 40, 192])
def test_every_tile_config(params, monkeypatch, config, m):
    w, ws, bias = params
    ks, wn, nrw, ms = host._config(1, config)
    wcnt = 56 // (wn * nrw) * ks
    if wcnt * ms < 16 * host.m_tile_fragments(m) or wcnt * ms > 256 or (ms > 1 and wcnt % 8):
        pytest.skip("fewer workgroups than tokens, more than the epoch words, or a row-split pair off one XCD")
    monkeypatch.setitem(host.CONFIG_BY_MT, host.m_tile_fragments(m), config)
    host._WORK.clear()
    x = torch.randn(m, H, device="cuda").to(torch.bfloat16)
    lg = router_logits(x, ws, shuffled=True)
    check_logits(lg, logits_reference(x, w))
    weights, ids = router_sigmoid_bias_top16_e896_h7168(x, ws, bias, shuffled=True)
    aw, aids = aiter_top16(lg, bias)
    assert torch.equal(ids, aids) and same_bits(weights, aw)


def test_layouts_agree(params):
    w, ws, bias = params
    x = torch.randn(80, H, device="cuda").to(torch.bfloat16)
    a = router_sigmoid_bias_top16_e896_h7168(x, w, bias)
    b = router_sigmoid_bias_top16_e896_h7168(x, ws, bias, shuffled=True)
    assert same_bits(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("m", [1, 7, 64, 192])
def test_top16_random(m):
    g = torch.Generator(device="cuda").manual_seed(m)
    check_top16(torch.randn(m, E, device="cuda", generator=g) * 3, torch.randn(E, device="cuda", generator=g) * 0.1, 2.5)


def test_top16_ties():
    z = torch.zeros(8, E, device="cuda")
    bias = torch.zeros(E, device="cuda")
    check_top16(z, bias)
    bias[::3] = 0.25
    check_top16(z, bias)
    lg = z.clone()
    lg[:, [5, 300, 700, 895]] = 3
    check_top16(lg, torch.zeros(E, device="cuda"))
    coarse_bias = torch.randint(0, 4, (E,), device="cuda").float() * 0.1
    g = torch.Generator(device="cuda").manual_seed(1)
    check_top16(torch.randn(64, E, device="cuda", generator=g).round() * 0.5, coarse_bias)
    check_top16(torch.zeros(4, E, device="cuda"), coarse_bias)
    # the whole top-16 inside one lane's slots, and ties split between lanes
    lg = torch.full((4, E), -20.0, device="cuda")
    lg[:, [0, 1, 2, 3, 256, 257, 258, 259, 512, 513, 514, 515, 768, 769, 770, 771]] = 1.0
    check_top16(lg, torch.zeros(E, device="cuda"))
    lg[:, 4:8] = 1.0
    check_top16(lg, torch.zeros(E, device="cuda"))


def test_top16_edge_scores():
    g = torch.Generator(device="cuda").manual_seed(2)
    check_top16(torch.randn(32, E, device="cuda", generator=g), -5 + torch.randn(E, device="cuda", generator=g))
    lg = torch.randn(8, E, device="cuda", generator=g)
    lg[:, 890:] = 50
    check_top16(lg, torch.zeros(E, device="cuda"))
    check_top16(torch.randn(8, E, device="cuda", generator=g) * 30, torch.randn(E, device="cuda", generator=g))


def test_back_to_back_calls_repeat_bits(params):
    """The slab parity tag: calls of different M (same and other row tiles) back to back, new weights and activations each
    time, never read a word of an older call: every call's bits equal the first call of the same inputs."""
    _w, ws, bias = params
    g = torch.Generator(device="cuda").manual_seed(3)
    xs = {m: torch.randn(m, H, device="cuda", generator=g).to(torch.bfloat16) for m in (12, 17, 32, 33, 80, 192, 1)}
    first = {m: router_sigmoid_bias_top16_e896_h7168(x, ws, bias, shuffled=True) for m, x in xs.items()}
    torch.cuda.synchronize()
    order = [12, 32, 17, 12, 33, 192, 17, 1, 80, 32, 12, 12, 33, 80, 1, 192] * 12
    outs = [(m, router_sigmoid_bias_top16_e896_h7168(xs[m], ws, bias, shuffled=True)) for m in order]
    torch.cuda.synchronize()
    for m, (ow, oi) in outs:
        assert torch.equal(oi, first[m][1]) and same_bits(ow, first[m][0]), m
