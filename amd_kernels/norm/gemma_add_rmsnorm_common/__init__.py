# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Residual add + Gemma RMSNorm entries: the one-GPU MXFP8 norm for the next GEMM (``add_rmsnorm.py``) and the TP
communicators of a decoder layer's two norm tails (``allreduce.py``); each reads its shape from its arguments and
routes to the shape package (``_ROUTES`` / ``_MLP``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .add_rmsnorm import (
    gemma_add_rmsnorm_mxfp8_for,
    gemma_add_rmsnorm_mxfp8_for_blocker,
    gemma_add_rmsnorm_mxfp8_for_shapes,
    warm_up_gemma_add_rmsnorm_mxfp8_for,
)
from .allreduce import (
    ar_norm_dense_mlp,
    ar_norm_dense_mlp_blocker,
    decoder_ar_norm,
    decoder_ar_norm_blocker,
    decoder_ar_norm_shapes,
    decoder_ar_norm_world_blocker,
    prepare_ar_norm_dense_mlp,
    prepare_decoder_ar_norm,
)

__all__ = [
    "ar_norm_dense_mlp",
    "ar_norm_dense_mlp_blocker",
    "available",
    "decoder_ar_norm",
    "decoder_ar_norm_blocker",
    "decoder_ar_norm_shapes",
    "decoder_ar_norm_world_blocker",
    "gemma_add_rmsnorm_mxfp8_for",
    "gemma_add_rmsnorm_mxfp8_for_blocker",
    "gemma_add_rmsnorm_mxfp8_for_shapes",
    "prepare_ar_norm_dense_mlp",
    "prepare_decoder_ar_norm",
    "warm_up_gemma_add_rmsnorm_mxfp8_for",
]
