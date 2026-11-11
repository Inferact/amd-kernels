# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The activation quantizer ``mxfp8_linear`` launches (FlyDSL ``mxfp8_quant``, every row count) against the torch
reference ``quantize_mxfp8``, bit for bit: row-major codes (``mxfp8_e4m3_quantize_aiter``) and the decode rows'
(16, 16) codes (``mxfp8_e4m3_quantize_shuffled``, padding rows zero), E8M0 scales in AITER's shuffled layout."""

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8

from .test_gemm import require_gfx950

ROWS = (1, 2, 15, 16, 17, 33, 100, 575, 576, 961, 1023, 1024, 2048, 4095, 4096)


def aiter_scale_offsets(rows: int, k: int, device) -> torch.Tensor:
    """Flat offsets of scale (row, block) in AITER's shuffled layout
    [rows // 32, K // 256, KLane 4, NLane 16, KPack 2, MPack 2], row-major order of (row, block)."""
    r = torch.arange(rows, device=device).view(-1, 1)
    c = torch.arange(k // 32, device=device).view(1, -1)
    m1, mpack, mlane = r // 32, (r % 32) // 16, r % 16
    k1, kpack, klane = c // 8, (c % 8) // 4, c % 4
    return (((((m1 * (k // 256) + k1) * 4 + klane) * 16 + mlane) * 2 + kpack) * 2 + mpack).view(-1)


@pytest.mark.parametrize("k", [2048, 3072, 6144])
@pytest.mark.parametrize("rows", ROWS)
@torch.inference_mode()
def test_row_major(rows, k):
    require_gfx950()
    torch.manual_seed(rows * 7 + k)
    x = (torch.randn((rows, k), device="cuda") * torch.logspace(-3, 3, k // 32, device="cuda").repeat_interleave(32)).bfloat16()
    x[0, :32] = 0  # an all-zero block
    codes, scales = G.mxfp8_e4m3_quantize_aiter(x)
    ref_c, ref_s = quantize_mxfp8(x)
    assert torch.equal(codes.view(torch.uint8), ref_c.view(torch.uint8)), int((codes.view(torch.uint8) != ref_c.view(torch.uint8)).sum())
    got_s = scales.view(-1)[aiter_scale_offsets(rows, k, x.device)]
    assert torch.equal(got_s, ref_s.view(-1)), int((got_s != ref_s.view(-1)).sum())


@pytest.mark.parametrize("tile", [16, 32, 64])
@pytest.mark.parametrize("rows", ROWS[:9])
@torch.inference_mode()
def test_shuffled(rows, tile):
    from aiter.ops.shuffle import shuffle_weight

    require_gfx950()
    k = 6144
    torch.manual_seed(rows * 11 + tile)
    x = (torch.randn((rows, k), device="cuda") * 4).bfloat16()
    codes, scales = G.mxfp8_e4m3_quantize_shuffled(x, tile)
    padded = (rows + tile - 1) // tile * tile
    ref_c, ref_s = quantize_mxfp8(x)
    ref_full = torch.zeros((padded, k), dtype=torch.float8_e4m3fn, device=x.device)
    ref_full[:rows] = ref_c
    want = shuffle_weight(ref_full, layout=(16, 16))
    assert torch.equal(codes.view(torch.uint8), want.view(torch.uint8)), int((codes.view(torch.uint8) != want.view(torch.uint8)).sum())
    got_s = scales.view(-1)[aiter_scale_offsets(rows, k, x.device)]
    assert torch.equal(got_s, ref_s.view(-1)), int((got_s != ref_s.view(-1)).sum())
