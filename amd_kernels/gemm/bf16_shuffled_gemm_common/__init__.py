# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 GEMM entries over a wide weight in its load-time shuffled layout: the op reads (N, K) from the weight and
routes to the shape package (``_ROUTES`` of ``gemm.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .gemm import (
    bf16_shuffled_gemm,
    bf16_shuffled_gemm_blocker,
    bf16_shuffled_gemm_shapes,
    bf16_shuffled_gemm_shuffle_weight_,
    bf16_shuffled_gemm_weight_blocker,
    prepare_bf16_shuffled_gemm_weight,
    warm_up_bf16_shuffled_gemm,
)

__all__ = [
    "available",
    "bf16_shuffled_gemm",
    "bf16_shuffled_gemm_blocker",
    "bf16_shuffled_gemm_shapes",
    "bf16_shuffled_gemm_shuffle_weight_",
    "bf16_shuffled_gemm_weight_blocker",
    "prepare_bf16_shuffled_gemm_weight",
    "warm_up_bf16_shuffled_gemm",
]
