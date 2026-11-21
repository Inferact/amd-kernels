# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16_latent_down_mxfp4_quant_k7168_n3584 against its reference, both weight layouts, every K split."""
import pytest
import torch

from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584 import (bf16_latent_down_mxfp4_quant_k7168_n3584,
                                                                       shuffle_weight, unshuffle_weight)
from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584 import host
from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584.reference import (K, N, down_reference,
                                                                                  quant_reference)

MS = [1, 2, 5, 12, 15, 16, 17, 24, 33, 40, 48, 65, 80, 96, 112, 127, 128, 129, 160, 176, 191, 192, 193, 300]


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return {False: w, True: shuffle_weight(w)}


def u8(t):
    return t.view(torch.uint8)


def check(x, wk, shuffled, w):
    m = x.shape[0]
    y = torch.full((m, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    q, s = bf16_latent_down_mxfp4_quant_k7168_n3584(x, wk, shuffled=shuffled, out_bf16=y)
    ref = down_reference(x, w)
    cos = torch.nn.functional.cosine_similarity(y.double(), ref.double(), dim=1)
    assert bool(torch.all(cos >= 0.99999)), cos.min().item()
    qr, sr = quant_reference(y)
    assert torch.equal(u8(q), qr) and torch.equal(u8(s), sr)
    return q, s, y


def test_shuffle_is_a_permutation(weights):
    assert torch.equal(unshuffle_weight(weights[True]).view(torch.int16), weights[False].view(torch.int16))


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("m", MS)
def test_reference(weights, shuffled, m):
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    q, s, y = check(x, weights[shuffled], shuffled, weights[False])
    q2, s2 = bf16_latent_down_mxfp4_quant_k7168_n3584(x, weights[shuffled], shuffled=shuffled)
    assert torch.equal(u8(q), u8(q2)) and torch.equal(u8(s), u8(s2))


@pytest.mark.parametrize("m", [3, 40, 192])
def test_placements_agree(weights, monkeypatch, m):
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    outs = []
    for local in (False, True):
        monkeypatch.setitem(host._XCD_LOCAL, torch.cuda.current_device(), local)
        host._WORK.clear()
        outs.append(bf16_latent_down_mxfp4_quant_k7168_n3584(x, weights[True], shuffled=True))
    assert torch.equal(u8(outs[0][0]), u8(outs[1][0])) and torch.equal(u8(outs[0][1]), u8(outs[1][1]))


def test_probe_finds_one_xcc_per_tile():
    assert host._xcd_local(torch.device("cuda", torch.cuda.current_device()))


def test_layouts_agree(weights):
    x = torch.randn(80, K, device="cuda").to(torch.bfloat16)
    a = bf16_latent_down_mxfp4_quant_k7168_n3584(x, weights[False])
    b = bf16_latent_down_mxfp4_quant_k7168_n3584(x, weights[True], shuffled=True)
    assert torch.equal(u8(a[0]), u8(b[0])) and torch.equal(u8(a[1]), u8(b[1]))


def test_quant_matches_aiter(weights):
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    x = torch.randn(65, K, device="cuda").to(torch.bfloat16)
    q, s, y = check(x, weights[False], False, weights[False])
    qa, sa = per_1x32_mx_quant_hip(y, quant_dtype=dtypes.fp4x2, shuffle=False)
    assert torch.equal(u8(q), u8(qa)) and torch.equal(u8(s), u8(sa))


def test_quant_edge_values():
    g = torch.Generator(device="cuda").manual_seed(1)
    y = torch.randn(64, N, device="cuda", generator=g) * torch.exp(torch.randn(64, 1, device="cuda", generator=g) * 3)
    y = y.to(torch.bfloat16)
    y[0], y[1], y[2, :32] = 0, 1e-30, 6.0
    y[3, 5], y[4] = -0.0, 3e38
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    qr, sr = quant_reference(y)
    qa, sa = per_1x32_mx_quant_hip(y, quant_dtype=dtypes.fp4x2, shuffle=False)
    assert torch.equal(u8(qa), qr) and torch.equal(u8(sa), sr)
