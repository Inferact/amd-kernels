# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Clamped SiLU-mul of a gate_up row (2 x 2304) and its MXFP8 quant in one launch, FlyDSL (gfx950)."""

from .host import clamp_silu_mul_mxfp8, warmup_clamp_silu_mul_mxfp8
from .reference import clamp_silu_mul_mxfp8_reference, clamp_silu_mul_reference, dequantize_mxfp8

__all__ = ["clamp_silu_mul_mxfp8", "clamp_silu_mul_mxfp8_reference", "clamp_silu_mul_reference", "dequantize_mxfp8",
           "warmup_clamp_silu_mul_mxfp8"]
