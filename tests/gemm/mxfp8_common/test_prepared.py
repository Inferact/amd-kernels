# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``prepare_*`` handles on one GPU: their outputs equal the direct calls bit for bit (the dense linear from the
checkpoint layout, the grouped GEMM's MXFP8 out on both sides of the decode package's row limit)."""

import pytest
import torch

from amd_kernels import gemm, routes
from amd_kernels.gemm.mxfp8_bmm_mxfp8_out_n1024_k4096 import mxfp8_bmm_mxfp8_out
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096 import grouped_gemm_mxfp8

pytestmark = pytest.mark.skipif(not routes.available(), reason="needs gfx950 with FlyDSL")
F8, U8 = torch.float8_e4m3fn, torch.uint8


def codes(*shape, gen):
    x = torch.randint(0, 256, shape, dtype=U8, device="cuda", generator=gen)
    return torch.where((x & 0x7F) == 0x7F, x & 0x80, x).view(F8)  # no NaN codes


def scales(*shape, gen):
    return torch.randint(118, 130, shape, dtype=U8, device="cuda", generator=gen)


@pytest.mark.parametrize("m", [1, 300])
@torch.inference_mode()
def test_mxfp8_linear_handle_equals_the_direct_call(m):
    gen = torch.Generator(device="cuda").manual_seed(7)
    w, ws = codes(1792, 5120, gen=gen), scales(1792, 160, gen=gen)
    h = gemm.prepare_mxfp8_linear(w, ws, max_num_tokens=512)
    x = torch.randn(m, 5120, device="cuda", generator=gen).to(torch.bfloat16)
    ref_w, ref_s = G.preshuffle_mxfp8_weight(w, ws)
    assert torch.equal(h(x), G.mxfp8_linear(x, ref_w, ref_s))


@pytest.mark.parametrize("m", [16, 1024, 1025, 4096])
@torch.inference_mode()
def test_grouped_mxfp8_out_handle_equals_the_packages(m):
    from aiter.ops.shuffle import shuffle_weight

    gen = torch.Generator(device="cuda").manual_seed(11)
    w = shuffle_weight(codes(8 * 1024, 4096, gen=gen), layout=(16, 16))
    ws = scales(8 * 32, 128, gen=gen)
    h = gemm.prepare_mxfp8_grouped_gemm(w, ws, 8, max_num_tokens=4096)
    x, xs = codes(m, 8 * 4096, gen=gen), scales(m, 8 * 128, gen=gen)
    got = h(x, xs)
    direct = grouped_gemm_mxfp8 if m <= 1024 else mxfp8_bmm_mxfp8_out
    want = direct(x, xs, w, ws.view(8, 32, 128))
    assert torch.equal(got[0].view(U8), want[0].view(U8)) and torch.equal(got[1], want[1])
