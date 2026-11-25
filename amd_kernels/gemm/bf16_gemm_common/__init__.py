# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 GEMM entries (``host._ROUTES``, ``host._SPLIT``): ``bf16_gemm`` (lm_head) and ``bf16_gemm_f32_bf16`` (fp32 +
bf16 column blocks in one GEMM)."""

from amd_kernels.routes import available

from .host import (
    bf16_gemm,
    bf16_gemm_blocker,
    bf16_gemm_weight_blocker,
    bf16_gemm_f32_bf16,
    bf16_gemm_f32_bf16_blocker,
    bf16_gemm_f32_bf16_shapes,
    bf16_gemm_f32_bf16_shuffle_weight,
    bf16_gemm_shapes,
    bf16_gemm_shuffle_weight,
    bf16_gemm_shuffle_weight_,
    prepare_bf16_gemm,
    warmup_bf16_gemm,
)

__all__ = [
    "available",
    "bf16_gemm",
    "bf16_gemm_blocker",
    "bf16_gemm_weight_blocker",
    "bf16_gemm_f32_bf16",
    "bf16_gemm_f32_bf16_blocker",
    "bf16_gemm_f32_bf16_shapes",
    "bf16_gemm_f32_bf16_shuffle_weight",
    "bf16_gemm_shapes",
    "bf16_gemm_shuffle_weight",
    "bf16_gemm_shuffle_weight_",
    "prepare_bf16_gemm",
    "warmup_bf16_gemm",
]
