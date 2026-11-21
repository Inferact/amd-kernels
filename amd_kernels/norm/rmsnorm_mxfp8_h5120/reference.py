# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``rmsnorm_mxfp8``: the RMSNorm of a hidden-size row, then its MXFP8 quantization.

    y = bf16(x * rsqrt(mean(x^2) + eps) * w)             5120 columns, fp32 math, one bf16 rounding

``y`` is quantized from the bf16 value (never the fp32 one) as the served Triton quantizer (``mxfp8_e4m3_quantize``):
per 32 columns the E8M0 byte ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)`` and e4m3 codes of ``y * 2^(127 - byte)``,
round to nearest even; codes row-major ``[m, 5120]``, scales row-major ``[m, 160]`` (vLLM's ``QuantizedActivation``
data and scale).

Tolerance: the kernel's square sum is its own fixed tree and its rsqrt the hardware's, so about 1e-5 of the bf16
values of ``y`` differ from torch's by an ulp, and a few codes with them; the tests bound that and check the
dequantized rows' cosine >= 0.99999 and that two runs match bit for bit.
"""

import torch

H = 5120


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).bfloat16()


def quantize_mxfp8_reference(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[m, K]`` -> e4m3 codes ``[m, K]`` and E8M0 bytes ``[m, K / 32]``."""
    m, k = y.shape
    blocks = y.float().reshape(m, k // 32, 32)
    amax = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    sb = torch.clamp(torch.ceil(torch.log2(amax / 448.0)) + 127.0, 0.0, 254.0)
    codes = (blocks * torch.exp2(127.0 - sb).unsqueeze(-1)).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), sb.to(torch.uint8)


def rmsnorm_mxfp8_reference(x, weight, eps):
    """(codes, scales) of ``x`` [m, 5120]."""
    return quantize_mxfp8_reference(rmsnorm_reference(x, weight, eps))


def dequantize_mxfp8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scale.float() - 127.0).unsqueeze(-1)).reshape(m, k)
