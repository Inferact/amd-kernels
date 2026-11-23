# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``prepare_mxfp8_linear``: a linear's MXFP8 weight bound in the GEMMs' layout and compiled, from the checkpoint's
layout (e4m3 [N, K] with e8m0 scales per 1 x 32 or 32 x 32 block)."""

import torch

from amd_kernels.routes import Prepared

from . import gemm as G


def row_scales(weight: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """e8m0 scales [N, K / 32] (uint8) from per-row [N, K / 32] or 32 x 32 block [N / 32, K / 32] scales."""
    n, k = weight.shape
    scale = weight_scale.view(torch.uint8)
    if tuple(scale.shape) == ((n + 31) // 32, (k + 31) // 32) and n > 1:
        scale = scale.repeat_interleave(32, dim=0)
    return scale[:n, : k // 32].contiguous()


def preshuffled(weight, weight_scale, weight_preshuffled=None):
    """``(weight, scales)`` in the GEMMs' layout (AITER's (16, 16) preshuffle, shuffled e8m0 scales); a
    ``weight_preshuffled`` already in that layout is used as is (no copy)."""
    scale = row_scales(weight, weight_scale)
    if weight_preshuffled is None:
        return G.preshuffle_mxfp8_weight(weight, scale)
    from aiter.ops.shuffle import shuffle_scale_a16w4

    return weight_preshuffled, shuffle_scale_a16w4(scale, 1, False)


class Mxfp8Linear(Prepared):
    """``x @ W^T`` for one preshuffled weight: ``h(x)`` on a bf16 / fp16 activation, ``h(codes, scales)`` on row-major
    MXFP8 codes and e8m0 scales."""

    op = "mxfp8_linear"

    def __init__(self, weight, weight_scale, out_dtype):
        self.weight, self.weight_scale, self.out_dtype = weight, weight_scale, out_dtype
        self.n, self.k = weight.shape

    def blocker(self, x, x_scale=None, out=None) -> str | None:
        return G._rows_blocker(x.shape[0], self.n, self.k, self.out_dtype, x_scale is not None)

    def __call__(self, x, x_scale=None, out=None):
        if x_scale is None:
            return G.mxfp8_linear(x, self.weight, self.weight_scale, self.out_dtype, out=out)
        return G.mxfp8_gemm(x, x_scale, self.weight, self.weight_scale, self.out_dtype, out=out)


def prepare_mxfp8_linear(weight, weight_scale, *, out_dtype: torch.dtype = torch.bfloat16, weight_preshuffled=None,
                         max_num_tokens: int = 1024) -> Mxfp8Linear | str:  # fmt: skip
    """The bound linear, compiled for 1..``max_num_tokens`` rows, or the reason no shape folder serves it.

    Args:
        weight: e4m3 [N, K] in the checkpoint's row-major layout.
        weight_scale: e8m0 scales per 1 x 32 ([N, K / 32]) or 32 x 32 ([N / 32, K / 32]) block.
        weight_preshuffled: the same weight already in AITER's (16, 16) preshuffle (used as is, no copy).
    """
    if weight.dim() != 2 or weight.dtype != G.MXFP8_VALUE_DTYPE:
        return f"mxfp8_linear: weight {tuple(weight.shape)} {weight.dtype} is not e4m3 [N, K]"
    n, k = weight.shape
    if (n, k) not in G.tuned_shapes(out_dtype):
        return G._rows_blocker(1, n, k, out_dtype, False) or f"mxfp8_linear: N {n} x K {k} has no tuned rows"
    w, s = preshuffled(weight, weight_scale, weight_preshuffled)
    G.warmup_mxfp8_gemm_weight(w, s, max_num_tokens, out_dtype)
    return Mxfp8Linear(w, s, out_dtype)
