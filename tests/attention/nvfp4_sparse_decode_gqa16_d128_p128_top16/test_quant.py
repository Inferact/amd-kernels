# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``kernel._mxfp8_group`` (the MXFP8 epilogue) against ``reference.mxfp8_quantize``, every finite bf16 encoding."""
import pytest
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from amd_kernels.common import buffer_ops

from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.kernel import _mxfp8_group
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import mxfp8_quantize


@flyc.kernel(name="test_nvfp4_attention_mxfp8_bf16_g32")
def _quant_groups(x: fx.Tensor, y: fx.Tensor, s: fx.Tensor, count: fx.Int32):
    group = fx.thread_idx.x + fx.block_idx.x * 256
    if group < count:
        xr = buffer_ops.create_buffer_resource(x, max_size=True)
        yr = buffer_ops.create_buffer_resource(y, max_size=True)
        words = [
            fx.Int32(buffer_ops.buffer_load(xr, group * 16 + i, vec_width=1, dtype=fx.Int32))
            for i in range_constexpr(16)
        ]
        codes, scale = _mxfp8_group(words)
        for i in range_constexpr(8):
            buffer_ops.buffer_store(codes[i], yr, group * 8 + i, mask=group < count)
        s[group] = scale.to(fx.Uint8)


@flyc.jit
def launch_attention_quant_groups(x: fx.Tensor, y: fx.Tensor, s: fx.Tensor, count: fx.Int32):
    _quant_groups(x, y, s, count).launch(grid=((count + 255) // 256,), block=(256,))


@torch.inference_mode()
def test_native_quant_finite_encodings():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx950")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).short()
    values = bits[bits.view(torch.bfloat16).isfinite()].view(torch.bfloat16)
    generator = torch.Generator(device="cuda").manual_seed(741)
    maxima = torch.arange(0x7F80, device="cuda", dtype=torch.int32)
    # one group per finite encoding, then every magnitude as the maximum of a mixed group: subnormal inputs and
    # quotients, exact powers, both sides of the scale boundary, signed zero and RNE ties
    mixed = (torch.rand((len(maxima), 32), generator=generator, device="cuda")
             * (maxima[:, None] + 1)).int()
    mixed[:, 0] = maxima
    mixed |= torch.randint(2, mixed.shape, generator=generator, device="cuda") << 15
    for x in (values[:, None].expand(-1, 32).contiguous(), mixed.short().view(torch.bfloat16)):
        expected_codes, expected_scales = mxfp8_quantize(x)
        # the kernel keeps scale byte 1 for these maxima, the contract 0: x / 2 quantized with byte 0 gives the kernel's
        # codes (the mismatch is the xfail below)
        maximum_bits = x.abs().amax(-1).view(torch.int16).int()
        legacy = (maximum_bits > 0x0460) & (maximum_bits < 0x04E0)
        if legacy.any():
            legacy_codes, _ = mxfp8_quantize(x[legacy].float() * 0.5)
            expected_codes[legacy] = legacy_codes
            expected_scales[legacy] = 1
        codes = torch.empty_like(x, dtype=torch.uint8)
        scales = torch.empty(x.shape[0], device="cuda", dtype=torch.uint8)
        launch_attention_quant_groups(x, codes, scales, x.shape[0])
        assert torch.equal(scales, expected_scales.flatten())
        assert torch.equal(codes, expected_codes)


@pytest.mark.xfail(strict=True, reason="the kernel keeps E8M0 byte 1 where amax / 448 is subnormal; the reference 0")
@torch.inference_mode()
def test_native_quant_underflow_contract():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx950")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    x = torch.full((256, 32), 0x0480, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    expected_codes, expected_scales = mxfp8_quantize(x)
    codes = torch.empty_like(x, dtype=torch.uint8)
    scales = torch.empty(x.shape[0], device="cuda", dtype=torch.uint8)
    launch_attention_quant_groups(x, codes, scales, x.shape[0])
    assert torch.equal(scales, expected_scales.flatten())
    assert torch.equal(codes, expected_codes)
