# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Tensor-parallel MXFP4 MoE layer as one persistent kernel (MegaMoE TP), and SpRsNorm (README)."""

from .layer import (
    COMM_MODES,
    ERR_BIGM,
    ActivationType,
    MegaMoeTP,
    MegaMoeTPConfig,
    MegaMoeTPLDSError,
    mega_moe_tp_supported,
)
from .layout import shuffle_scale, shuffle_weight
from .sp_rs_norm import SpRsNorm

__all__ = [
    "COMM_MODES",
    "ERR_BIGM",
    "ActivationType",
    "MegaMoeTP",
    "MegaMoeTPConfig",
    "MegaMoeTPLDSError",
    "SpRsNorm",
    "mega_moe_tp_supported",
    "shuffle_scale",
    "shuffle_weight",
]
