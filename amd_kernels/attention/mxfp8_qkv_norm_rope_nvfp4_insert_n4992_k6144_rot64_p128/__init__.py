# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 QKV MXFP8 GEMM (N 4992, K 6144) + split-K reduce + q / k / index norm + RoPE (rotary 64) + NVFP4 K / V /
index-K inserts into 128-token pages, one launch at 1-768 rows (FlyDSL, gfx950)."""

from .host import mxfp8_qkv_norm_rope_nvfp4_insert

__all__ = ["mxfp8_qkv_norm_rope_nvfp4_insert"]
