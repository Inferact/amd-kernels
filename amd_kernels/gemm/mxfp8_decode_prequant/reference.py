# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch references of the decode GEMM on pre-quantized activations and of its swiglu + MXFP8 epilogue.

``mxfp8_gemm_reference``: ``out = A @ W^T`` with A as E4M3 codes [M, K] and row-major E8M0 scales [M, K / 32] (a bf16
activation is quantized first by ``quantize_mxfp8``: ``e = clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254)``,
``code = e4m3fn(x * 2^(127 - e))``, the vLLM Triton quantizer's bytes), W as row-major E4M3 codes [N, K] and E8M0
scales [N, K / 32]; both dequantized exactly, fp32 sums, one bf16 rounding. The kernels differ only by accumulation
order and that rounding: cosine >= ``COS_MIN``, relative L2 <= ``REL_L2_MAX``, bitwise repeats.

``mxfp8_gemm_swiglu_reference``: the gate_up GEMM (rows [gate; up]) rounded to bf16, then the served
``SiluAndMulWithClamp`` (alpha 1, beta 0): ``g = min(gate, limit)``, ``u = clamp(up, -limit, limit) + 0``,
``bf16(bf16(silu(g)) * u)``, then ``quantize_mxfp8`` of that bf16 (codes [M, I], scales [M, I / 32]). A different
accumulation order can move a value across a rounding boundary: the dequantized result meets
``SWIGLU_COS_MIN`` / ``SWIGLU_REL_L2_MAX``.
"""

import torch

E4M3_MAX = 448.0
BLOCK = 32
COS_MIN = 0.99999
REL_L2_MAX = 1e-3
SWIGLU_COS_MIN = 0.99999
SWIGLU_REL_L2_MAX = 5e-3


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[R, K] -> E4M3 codes [R, K] and E8M0 exponents [R, K / 32], row-major."""
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (amax / E4M3_MAX).log2().ceil().add(127).clamp(0, 254)
    codes = (blocks * torch.exp2(127 - exponent).unsqueeze(-1)).reshape(rows, k)
    return codes.to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def dequantize_mxfp8(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """E4M3 codes and E8M0 exponents -> fp32 values (exact)."""
    rows, k = codes.shape
    scale = torch.exp2(exponents.float() - 127).unsqueeze(-1)
    return (codes.float().reshape(rows, k // BLOCK, BLOCK) * scale).reshape(rows, k)


def mxfp8_gemm_reference(codes, exponents, weight_codes, weight_exponents) -> torch.Tensor:
    """The contract's ``A @ W^T`` for a quantized activation, bf16."""
    return (dequantize_mxfp8(codes, exponents) @ dequantize_mxfp8(weight_codes, weight_exponents).T).to(torch.bfloat16)


def swiglu_clamp_reference(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    i = gate_up.shape[-1] // 2
    gate = torch.clamp(gate_up[..., :i].float(), max=limit)
    up = torch.clamp(gate_up[..., i:].float(), min=-limit, max=limit) + 0.0
    return (torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16()


def mxfp8_gemm_swiglu_reference(codes, exponents, weight_codes, weight_exponents, limit: float = 10.0):
    """(codes [M, I], exponents [M, I / 32]) of the quantized swiglu of the bf16 gate_up GEMM."""
    gate_up = mxfp8_gemm_reference(codes, exponents, weight_codes, weight_exponents)
    return quantize_mxfp8(swiglu_clamp_reference(gate_up, limit))


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Cosine similarity and relative L2 error of ``got`` against ``ref``."""
    g, r = got.float().flatten(), ref.float().flatten()
    return {
        "cos": torch.nn.functional.cosine_similarity(g, r, dim=0).item(),
        "rel_l2": ((g - r).norm() / r.norm()).item(),
    }
