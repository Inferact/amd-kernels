# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMS norm + RoPE + KV insert entries: one per op (one layer's qkv rows, the context rows of every layer), each
reads its shape from the tensors and routes to the shape package (``_ROUTES`` of each module); an uncovered call
gets a reason from ``*_blocker``."""

from .device import available
from .multilayer import (
    kv_norm_rope_insert_multilayer,
    kv_norm_rope_insert_multilayer_blocker,
    kv_norm_rope_insert_multilayer_kernel,
    kv_norm_rope_insert_multilayer_shapes,
    kv_norm_rope_insert_multilayer_unsupported_reason,
)
from .qkv import (
    norm_rope_insert_rope_blocker,
    qkv_norm_rope_insert,
    qkv_norm_rope_insert_blocker,
    qkv_norm_rope_insert_kernel,
    qkv_norm_rope_insert_shapes,
    qkv_norm_rope_insert_unsupported_reason,
    rope_unsupported_reason,
)

__all__ = [
    "available",
    "kv_norm_rope_insert_multilayer",
    "kv_norm_rope_insert_multilayer_blocker",
    "kv_norm_rope_insert_multilayer_kernel",
    "kv_norm_rope_insert_multilayer_shapes",
    "norm_rope_insert_rope_blocker",
    "qkv_norm_rope_insert",
    "qkv_norm_rope_insert_blocker",
    "qkv_norm_rope_insert_kernel",
    "qkv_norm_rope_insert_shapes",
]
