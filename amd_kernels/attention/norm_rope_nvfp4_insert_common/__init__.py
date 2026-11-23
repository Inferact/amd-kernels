# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse-layer attention input entries (q / k / index norms + RoPE + NVFP4 inserts, from the QKV rows or the
QKV GEMM's MXFP8 input): each reads its shape from the tensors and routes to the shape package (``_ROUTES`` of
``qkv.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .qkv import (
    mxfp8_gemm_qkv_norm_rope_nvfp4_insert,
    mxfp8_gemm_qkv_norm_rope_nvfp4_insert_blocker,
    norm_rope_nvfp4_insert_layer_blocker,
    qkv_norm_rope_nvfp4_insert,
    qkv_norm_rope_nvfp4_insert_blocker,
    qkv_norm_rope_nvfp4_insert_shapes,
)

__all__ = [
    "available",
    "mxfp8_gemm_qkv_norm_rope_nvfp4_insert",
    "mxfp8_gemm_qkv_norm_rope_nvfp4_insert_blocker",
    "norm_rope_nvfp4_insert_layer_blocker",
    "qkv_norm_rope_nvfp4_insert",
    "qkv_norm_rope_nvfp4_insert_blocker",
    "qkv_norm_rope_nvfp4_insert_shapes",
]
