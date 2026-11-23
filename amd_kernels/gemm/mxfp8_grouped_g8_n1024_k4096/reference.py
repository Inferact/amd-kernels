# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference for the grouped MXFP8 GEMM: the contract of both entries.

Contract (``out[m, g, :] = x[m, g, :] @ W[g]^T``, G 8, N 1024, K 4096):

- ``x``: E4M3 codes ``[M, G * K]`` with E8M0 exponents ``[M, G * K // 32]`` (1 x 32 blocks along K, row-major),
  quantized by the producer.
- ``W``: E4M3 codes ``[G * N, K]`` with E8M0 exponents ``[G, N // 32, K // 32]`` (32 x 32 blocks, the checkpoint's).
- Both operands are dequantized exactly, the product accumulates in FP32 per group, and the output is rounded once
  to BF16 ``[M, G * N]``. A kernel differs from this only by accumulation order: every output is finite with
  cosine similarity >= ``COS_MIN`` and relative L2 error <= ``REL_L2_MAX``. Repeated launches on the same inputs
  are bitwise identical.
- The MXFP8 entry returns ``quantize_mxfp8`` (1 x 32 blocks along G * N, ``mxfp8_common.reference``) of that BF16
  output: codes ``[M, G * N]`` and E8M0 exponents ``[M, G * N // 32]`` row-major, the pre-quantized activation of a
  ``[G * N, K']`` MXFP8 GEMM. Group boundaries fall on block boundaries, so a block never spans two groups. A value
  of the BF16 output that sits on an E4M3 rounding boundary may land one code away under another accumulation
  order, so the dequantized codes meet ``COS_MIN`` and ``QUANT_REL_L2_MAX`` against the reference's (measured
  1.2e-3 at most; with the same accumulation order the bytes are equal).
"""

import torch

from amd_kernels.gemm.mxfp8_common.reference import (
    COS_MIN,
    REL_L2_MAX,
    compare,
    dequantize_mxfp8,
    quantize_mxfp8,
)

G = 8
N = 1024
K = 4096
BLOCK = 32
QUANT_REL_L2_MAX = 5e-3

__all__ = [
    "BLOCK", "COS_MIN", "G", "K", "N", "QUANT_REL_L2_MAX", "REL_L2_MAX", "compare", "dequantize_mxfp8",
    "dequantize_weight",
    "grouped_gemm_mxfp8_reference", "grouped_gemm_reference", "quantize_mxfp8",
]


def dequantize_weight(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    """Row-major codes ``[G * N, K]`` and 32 x 32 exponents ``[G, N // 32, K // 32]`` -> FP32 ``[G, N, K]``."""
    scale = torch.exp2(exponents.to(torch.float32) - 127)
    scale = scale.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)
    return codes.view(G, N, K).to(torch.float32) * scale


def grouped_gemm_reference(
    codes: torch.Tensor, exponents: torch.Tensor, weight_codes: torch.Tensor, weight_exponents: torch.Tensor
) -> torch.Tensor:
    """BF16 ``[M, G * N]`` of the contract; the weight is row-major (not preshuffled)."""
    m = codes.shape[0]
    x = dequantize_mxfp8(codes.view(m * G, K), exponents.view(m * G, K // BLOCK)).view(m, G, K)
    w = dequantize_weight(weight_codes, weight_exponents)
    return torch.einsum("mgk,gnk->mgn", x, w).reshape(m, G * N).to(torch.bfloat16)


def grouped_gemm_mxfp8_reference(
    codes: torch.Tensor, exponents: torch.Tensor, weight_codes: torch.Tensor, weight_exponents: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 entry's contract: ``quantize_mxfp8`` of ``grouped_gemm_reference``."""
    return quantize_mxfp8(grouped_gemm_reference(codes, exponents, weight_codes, weight_exponents))
