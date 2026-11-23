# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Expert-parallel MXFP8 x MXFP4 routed experts (H 5120, I 2304, top 6) on 2 / 4 / 8 ranks: dispatch, GEMM1 +
clamp-SwiGLU + MXFP8 quant, GEMM2 + route weights and combine in one persistent launch per rank."""

from .host import ExpertParallel, ExpertWeights

__all__ = ["ExpertParallel", "ExpertWeights"]
