# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired gate/up MXFP8 GEMM + SwiGLU-OAI at 1-16 rows (dense MLP, TP2: N 12288, K 6144, I 6144)."""

from .host import mxfp8_gateup_swiglu_bf16, warmup_mxfp8_gateup_swiglu_bf16

__all__ = ["mxfp8_gateup_swiglu_bf16", "warmup_mxfp8_gateup_swiglu_bf16"]
