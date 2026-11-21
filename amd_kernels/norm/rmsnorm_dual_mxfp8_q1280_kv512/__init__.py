# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dual RMSNorm of a 1280-column q row (to MXFP8) and a 512-column kv row (to bf16) in one launch, FlyDSL (gfx950)."""

from .host import rmsnorm_dual_mxfp8, warmup_rmsnorm_dual_mxfp8
from .reference import dequantize_mxfp8, rmsnorm_dual_mxfp8_reference, rmsnorm_reference

__all__ = ["dequantize_mxfp8", "rmsnorm_dual_mxfp8", "rmsnorm_dual_mxfp8_reference", "rmsnorm_reference",
           "warmup_rmsnorm_dual_mxfp8"]
