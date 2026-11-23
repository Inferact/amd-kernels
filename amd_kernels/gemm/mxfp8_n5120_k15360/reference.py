# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``main_proj``: ``out = x @ W^T`` for the draft's input projection, ``x`` bf16 ``[M, 15360]`` (the
three target tap layers' hidden states side by side), ``W`` E4M3 codes ``[5120, 15360]`` with the checkpoint's 32 x 32
E8M0 block scales ``[160, 480]`` (one scale per 32 output rows and 32 K).

``x`` is quantized per 1 x 32 block along K: ``e = clamp(ceil(log2(max(amax, tiny) / 448)) + 127, 0, 254)`` (E8M0),
``code = e4m3fn(x * 2^(127 - e))``, round to nearest even, the multiplier applied as ``2^(127 - e)``. Both operands
are dequantized exactly, the product accumulates in FP32 and the result is rounded once to BF16. The kernels
differ from this only in accumulation order and split-K partial sums (summed in a fixed order): every output is finite
with cosine >= 0.99999 and relative L2 <= 1e-3; two runs match bit for bit.
"""

import torch

N, K = 5120, 15360
BLOCK = 32
E4M3_MAX = 448.0
COS_MIN = 0.99999
REL_L2_MAX = 1e-3


def expand_block_scales(block_scale: torch.Tensor) -> torch.Tensor:
    """``[N / 32, K / 32]`` block scales -> the per-row ``[N, K / 32]`` scales of the same weight."""
    return block_scale.repeat_interleave(BLOCK, dim=0).contiguous()


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[R, K]`` -> E4M3 codes and E8M0 exponents ``[R, K / 32]``, row-major."""
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    exponent = (amax / E4M3_MAX).log2().ceil().add(127).clamp(0, 254)
    codes = (blocks * torch.exp2(127 - exponent).unsqueeze(-1)).reshape(rows, k)
    return codes.to(torch.float8_e4m3fn), exponent.to(torch.uint8)


def dequantize(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    rows, k = codes.shape
    return (codes.float().reshape(rows, k // BLOCK, BLOCK) * torch.exp2(exponents.float() - 127).unsqueeze(-1)).reshape(rows, k)


def main_proj_reference(x: torch.Tensor, weight: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """bf16 ``[M, 5120]``: the module's contract for ``weight`` E4M3 ``[5120, 15360]`` and E8M0 ``block_scale``."""
    w = dequantize(weight, expand_block_scales(block_scale))
    return (dequantize(*quantize_mxfp8(x)) @ w.T).to(torch.bfloat16)


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    g, r = got.float().flatten(), ref.float().flatten()
    return {"cos": torch.nn.functional.cosine_similarity(g, r, dim=0).item(),
            "rel_l2": ((g - r).norm() / r.norm()).item()}
