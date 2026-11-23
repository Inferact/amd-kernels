# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP4 MoE block entries (routed experts + MXFP8 shared expert): the weight build reads its shape from the
checkpoint tensors and routes to the shape package (``_ROUTES`` of ``moe.py``), the op follows the package that built
its weights; an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .moe import (
    mxfp4_moe,
    mxfp4_moe_blocker,
    mxfp4_moe_set_dbo_query,
    mxfp4_moe_shapes,
    mxfp4_moe_weights,
    mxfp4_moe_weights_blocker,
    prepare_mxfp4_moe_weights,
    warm_up_mxfp4_moe,
)

__all__ = [
    "available",
    "mxfp4_moe",
    "mxfp4_moe_blocker",
    "mxfp4_moe_set_dbo_query",
    "mxfp4_moe_shapes",
    "mxfp4_moe_weights",
    "mxfp4_moe_weights_blocker",
    "prepare_mxfp4_moe_weights",
    "warm_up_mxfp4_moe",
]
