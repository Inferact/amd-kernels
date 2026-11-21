# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""amd_kernels.gemm.bf16_skinny_common against its fp64 reference on every served shape and row count."""
import pytest
import torch

from amd_kernels.gemm.bf16_skinny_common import SHAPES, bf16_linear, pack_weight, unpack_weight
from amd_kernels.gemm.bf16_skinny_common.reference import bf16_linear_reference

MS = [1, 5, 12, 15, 16, 17, 24, 40, 48, 50, 64, 65, 80, 96, 112, 128, 160, 176, 192, 193, 300]


def check_numeric(y, ref):
    yf, rf = y.double(), ref.double()
    cos = torch.nn.functional.cosine_similarity(yf, rf, dim=1)
    assert bool(torch.all(cos >= 0.99999)), cos.min().item()
    # the fp32 sum order can flip the final bf16 rounding of an element (one bf16 ulp, 2^-7 relative) and moves a
    # cancelling sum by the fp32 noise of the row (1e-5 of its largest element)
    assert bool(torch.all((yf - rf).abs() <= rf.abs() * 2.0**-7 + 1e-5 * rf.abs().amax(dim=1, keepdim=True)))
    assert float(((yf - rf).norm(dim=1) / rf.norm(dim=1)).median()) <= 1e-3


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    cache = {}

    def get(n, k):
        if (n, k) not in cache:
            w = (torch.randn(n, k, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
            cache[(n, k)] = (w, pack_weight(w))
        return cache[(n, k)]
    return get


def test_pack_is_a_permutation(weights):
    w, wp = weights(*SHAPES[0])
    assert torch.equal(unpack_weight(wp).view(torch.int16), w.view(torch.int16))
    scratch = torch.empty_like(w)
    assert unpack_weight(wp, scratch) is scratch
    assert torch.equal(scratch.view(torch.int16), w.view(torch.int16))


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"n{s[0]}_k{s[1]}")
@pytest.mark.parametrize("m", MS)
def test_reference(weights, shape, m):
    n, k = shape
    w, wp = weights(n, k)
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    y = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    bf16_linear(x, wp, y)
    check_numeric(y, bf16_linear_reference(x, w))
    first = y.clone()
    for _ in range(3):
        y.fill_(float("nan"))
        bf16_linear(x, wp, y)
        assert torch.equal(y.view(torch.int16), first.view(torch.int16))


def test_zero_rows(weights):
    n, k = SHAPES[0]
    _, wp = weights(n, k)
    assert bf16_linear(torch.empty(0, k, device="cuda", dtype=torch.bfloat16), wp).shape == (0, n)


@pytest.mark.parametrize("shape", SHAPES[:4], ids=lambda s: f"n{s[0]}_k{s[1]}")
def test_nonfinite_propagation(weights, shape):
    n, k = shape
    w, _ = weights(n, k)
    w = w.clone()
    w[0, 2] = float("nan")
    w[n - 1, 8] = float("inf")
    x = torch.randn(3, k, device="cuda").to(torch.bfloat16)
    x[0, 1] = float("nan")
    x[2].zero_()
    y = bf16_linear(x, pack_weight(w))
    ref = bf16_linear_reference(x, w)
    assert torch.equal(torch.isnan(y), torch.isnan(ref))
    assert torch.equal(torch.isinf(y), torch.isinf(ref))
