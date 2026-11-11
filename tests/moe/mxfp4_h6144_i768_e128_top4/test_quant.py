# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The chain input's two MXFP4 quantizers against ``reference.mxfp4_quant_ref`` bit for bit: ``quant.mxfp4_quant`` of
bf16 (one binade per 32-block from 2^-20 to 2^20, all-zero blocks, many values on e2m1 midpoints) and
``mxfp4_from_mxfp8`` of a gathered MXFP8 input (every e4m3 code but NaN, scales 2^-17 .. 2^17); three runs bitwise
equal, graph replay into 0xA5-filled outputs."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.mxfp4_from_mxfp8 import mxfp4_from_mxfp8
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.prefill_h6144_i768 import H
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import mxfp4_quant_ref

from .workload import make_mxfp8, replay_check, require_gfx950

SIZES = [1, 300, 4097, 32768]


@pytest.fixture(autouse=True)
def _gpu():
    require_gfx950()


def _bf16_blocks(m, gen):
    """bf16 [m, H] with a binade per 32-block from 2^-20 to 2^20 and some all-zero
    blocks; bf16's 8-bit mantissa puts many values exactly on e2m1 midpoints."""
    x = torch.randn((m, H // 32, 32), generator=gen)
    x *= torch.exp2(torch.randint(-20, 21, (m, H // 32, 1), generator=gen).float())
    x[torch.rand((m, H // 32), generator=gen) < 0.02] = 0.0
    return x.view(m, H).bfloat16().cuda()


def _check(fn, ref):
    q, s = fn()
    rq, rs = ref
    assert torch.equal(s, rs)
    assert torch.equal(q, rq)
    for _ in range(2):
        q2, s2 = fn()
        assert torch.equal(q2, q) and torch.equal(s2, s), "not bitwise reproducible"
    eager, replayed = replay_check(fn)
    for a, b in zip(eager, replayed):
        assert torch.equal(a, b), "graph replay differs"


@pytest.mark.parametrize("m", SIZES)
def test_mxfp4_quant_matches_reference(m):
    x = _bf16_blocks(m, torch.Generator().manual_seed(m))
    _check(lambda: mxfp4_quant(x), mxfp4_quant_ref(x))


@pytest.mark.parametrize("m", SIZES)
def test_mxfp4_from_mxfp8_matches_reference(m):
    codes, scales = make_mxfp8(m, seed=m)
    x = codes.view(torch.float8_e4m3fn).float() * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
    _check(lambda: mxfp4_from_mxfp8(codes, scales), mxfp4_quant_ref(x))
