# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMSNorm + MXFP8 entries (``host._ROUTES``, ``host._DUAL``)."""

from .host import (rmsnorm_dual_mxfp8, rmsnorm_dual_mxfp8_blocker, rmsnorm_dual_mxfp8_shapes, rmsnorm_mxfp8,
                   rmsnorm_mxfp8_blocker, rmsnorm_mxfp8_shapes, warmup_rmsnorm_dual_mxfp8, warmup_rmsnorm_mxfp8)

__all__ = ["rmsnorm_dual_mxfp8", "rmsnorm_dual_mxfp8_blocker", "rmsnorm_dual_mxfp8_shapes", "rmsnorm_mxfp8",
           "rmsnorm_mxfp8_blocker", "rmsnorm_mxfp8_shapes", "warmup_rmsnorm_dual_mxfp8", "warmup_rmsnorm_mxfp8"]
