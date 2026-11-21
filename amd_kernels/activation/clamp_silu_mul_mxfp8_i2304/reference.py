# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``clamp_silu_mul_mxfp8``.

The activation is the served ``SiluAndMulWithClamp`` (alpha 1, beta 0; its HIP op) on a gate_up row ``[gate | up]``::

    g = min(gate, limit); u = clamp(up, -limit, limit) + 0.0        (a -0 up reads as +0, as the HIP op)
    act = bf16(bf16(silu(g)) * u)           silu(g) = g / (1 + exp(-g)) in fp32

The quantizer then reads the bf16 ``act`` as the served Triton MXFP8 quantizer (``mxfp8_e4m3_quantize``): per 32 columns
the E8M0 byte ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)`` and e4m3 codes of ``act * 2^(127 - byte)``; codes
row-major ``[m, 2304]``, scales row-major ``[m, 72]``.

Tolerance: the kernel's exp may differ from torch's in the last fp32 bit, so ``act`` can differ by one bf16 ulp in a
few elements per million; the tests bound that and check per-row cosine >= 0.99999 of the dequantized rows.
"""

import torch

I = 2304


def clamp_silu_mul_reference(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    gate = torch.clamp(gate_up[..., :I].float(), max=limit)
    up = torch.clamp(gate_up[..., I:].float(), min=-limit, max=limit) + 0.0
    return (torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16()


def quantize_mxfp8_reference(act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[m, K]`` -> e4m3 codes ``[m, K]`` and E8M0 bytes ``[m, K / 32]``."""
    m, k = act.shape
    blocks = act.float().reshape(m, k // 32, 32)
    amax = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    sb = torch.clamp(torch.ceil(torch.log2(amax / 448.0)) + 127.0, 0.0, 254.0)
    codes = (blocks * torch.exp2(127.0 - sb).unsqueeze(-1)).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), sb.to(torch.uint8)


def clamp_silu_mul_mxfp8_reference(gate_up: torch.Tensor, limit: float):
    """(codes, scales)."""
    return quantize_mxfp8_reference(clamp_silu_mul_reference(gate_up, limit))


def dequantize_mxfp8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scale.float() - 127.0).unsqueeze(-1)).reshape(m, k)
