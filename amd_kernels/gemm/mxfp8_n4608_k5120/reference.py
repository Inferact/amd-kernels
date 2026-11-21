# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference of the N 4608 x K 5120 MXFP8 GEMM: the contract the entries of ``host.py`` meet.

``out = x @ W^T`` for a bf16 ``x`` [M, 5120] and a weight of E4M3 codes [N, K] with E8M0 per-32 scales [N, K / 32]
(the checkpoint's 32 x 32 block scales, one scale row per 32 weight rows). ``x`` is quantized per 1 x 32 block along
K: ``e = clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254)`` (E8M0), ``code = e4m3fn(x * 2^(127 - e))``, round
to nearest even (the vLLM Triton quantizer's bytes); pre-quantized activations arrive as the same codes and
row-major scales. Both operands are dequantized exactly, the product accumulates in fp32 and the output is rounded
once to bf16. The kernels differ from it only by accumulation order and that rounding: cosine >= ``COS_MIN``,
relative L2 <= ``REL_L2_MAX``; two launches on the same inputs match bit for bit.

``gate_up_swiglu_reference``: the GEMM rounded to bf16 (rows [gate; up], N = 2 I, I = 2304), then the served
``SiluAndMulWithClamp`` (alpha 1, beta 0): ``g = min(gate, limit)``, ``u = clamp(up, -limit, limit) + 0``,
``bf16(bf16(silu(g)) * u)``, then ``quantize_mxfp8`` of that bf16: codes [M, I], E8M0 bytes [M, I / 32]. A different
accumulation order can move a value across a rounding boundary: the dequantized result meets ``SWIGLU_COS_MIN`` /
``SWIGLU_REL_L2_MAX``, and is bit for bit the reference epilogue applied to the kernel's own bf16 sums.
"""

import torch

N = 4608
K = 5120
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


def mxfp8_linear_reference(x, weight_codes, weight_exponents, exponents=None) -> torch.Tensor:
    """The contract's ``x @ W^T`` for a row-major weight; ``x`` bf16, or (``exponents`` given) E4M3 codes."""
    codes, exponents = (x, exponents) if exponents is not None else quantize_mxfp8(x)
    return (dequantize_mxfp8(codes, exponents) @ dequantize_mxfp8(weight_codes, weight_exponents).T).to(torch.bfloat16)


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Cosine similarity and relative L2 error of ``got`` against ``ref``."""
    g, r = got.float().flatten(), ref.float().flatten()
    return {
        "cos": torch.nn.functional.cosine_similarity(g, r, dim=0).item(),
        "rel_l2": ((g - r).norm() / r.norm()).item(),
    }


def gate_up_swiglu_reference(x, weight_codes, weight_exponents, exponents=None, limit: float = 10.0):
    """(codes [M, I], exponents [M, I / 32]) of the quantized clamped SiLU-mul of the bf16 gate_up GEMM."""
    gate_up = mxfp8_linear_reference(x, weight_codes, weight_exponents, exponents)
    i = gate_up.shape[-1] // 2
    gate = torch.clamp(gate_up[:, :i].float(), max=limit)
    up = torch.clamp(gate_up[:, i:].float(), min=-limit, max=limit) + 0.0
    return quantize_mxfp8((torch.nn.functional.silu(gate).bfloat16().float() * up).bfloat16())
