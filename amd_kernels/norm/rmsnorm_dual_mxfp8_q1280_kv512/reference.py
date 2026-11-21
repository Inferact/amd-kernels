# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``rmsnorm_dual_mxfp8``.

Both norms are fp32 with a single bf16 rounding of the result, as the served Triton ``fused_q_kv_rmsnorm``::

    y = bf16(x * rsqrt(mean(x^2) + eps) * w)            q: 1280 columns, kv: 512 columns

The q output is then quantized from the bf16 value (never the fp32 one) as the served Triton MXFP8 quantizer
(``mxfp8_e4m3_quantize``): per 32 columns the E8M0 byte ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)`` and e4m3 codes
of ``y * 2^(127 - byte)``; codes row-major ``[m, 1280]``, scales row-major ``[m, 40]``.

Tolerance: the kernel's square sum is the served Triton norm's reduction tree (4 columns per thread, DPP wave fold,
warp tree), so its outputs equal the served ``fused_q_kv_rmsnorm`` + ``mxfp8_e4m3_quantize`` bit for bit. torch's sum
differs from it in the last fp32 bit, so against this reference about 3e-6 of the bf16 values differ by an ulp (2e-7
of the codes); the tests bound that and check per-row cosine >= 0.99999 of the dequantized rows.
"""

import torch

Q, KV = 1280, 512


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


def rmsnorm_dual_mxfp8_reference(q, kv, q_weight, kv_weight, eps):
    """(q codes, q scales, kv bf16)."""
    codes, scale = quantize_mxfp8_reference(rmsnorm_reference(q, q_weight, eps))
    return codes, scale, rmsnorm_reference(kv, kv_weight, eps)


def dequantize_mxfp8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scale.float() - 127.0).unsqueeze(-1)).reshape(m, k)
