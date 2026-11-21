# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 x MXFP4 routed experts, H 5120 I 2304 E 48 top 6: grouped GEMM1 + clamp-SwiGLU + MXFP8 quant, GEMM2 + route
weights + deterministic slot combine."""

from .host import ExpertWeights, mxfp4_experts, mxfp4_experts_rows

__all__ = ["ExpertWeights", "mxfp4_experts", "mxfp4_experts_rows"]
