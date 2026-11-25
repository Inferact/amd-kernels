# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gemma RMS norm + RoPE + e4m3 KV insert entries (the ``1 + weight`` norm contract, apart from
``norm_rope_insert_common``): each reads its shape from the tensors and routes to the shape package (``_ROUTES`` of
``qkv.py``); an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .qkv import (
    gemma_norm_rope_insert_rope_blocker,
    gemma_qkv_norm_rope_insert,
    gemma_qkv_norm_rope_insert_blocker,
    gemma_qkv_norm_rope_insert_kernel,
    gemma_qkv_norm_rope_insert_shapes,
)

__all__ = [
    "available",
    "gemma_norm_rope_insert_rope_blocker",
    "gemma_qkv_norm_rope_insert",
    "gemma_qkv_norm_rope_insert_blocker",
    "gemma_qkv_norm_rope_insert_kernel",
    "gemma_qkv_norm_rope_insert_shapes",
]
