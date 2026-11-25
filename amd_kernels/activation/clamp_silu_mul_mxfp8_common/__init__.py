# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Clamped SiLU-mul + MXFP8 entry (``host._ROUTES``)."""

from .host import (clamp_silu_mul_mxfp8, clamp_silu_mul_mxfp8_blocker, clamp_silu_mul_mxfp8_shapes,
                   warmup_clamp_silu_mul_mxfp8)

__all__ = ["clamp_silu_mul_mxfp8", "clamp_silu_mul_mxfp8_blocker", "clamp_silu_mul_mxfp8_shapes",
           "warmup_clamp_silu_mul_mxfp8"]
