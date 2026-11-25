# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse attention entries (decode, index + decode, prefill) and the layer's load-time state: each reads its
shape from the tensors (or the layer's head counts) and routes to the layer package (``_*`` Routes of ``layer.py``);
an uncovered call gets a reason from ``*_blocker``."""

from amd_kernels.routes import available

from .layer import (
    Nvfp4SparseLayer,
    nvfp4_index_sparse_decode,
    nvfp4_index_sparse_decode_blocker,
    nvfp4_sparse_attention_shapes,
    nvfp4_sparse_decode,
    nvfp4_sparse_decode_blocker,
    nvfp4_sparse_layer,
    nvfp4_sparse_layer_blocker,
    nvfp4_sparse_prefill,
    nvfp4_sparse_prefill_blocker,
    prepare_nvfp4_sparse_layer,
)

__all__ = [
    "Nvfp4SparseLayer",
    "available",
    "nvfp4_index_sparse_decode",
    "nvfp4_index_sparse_decode_blocker",
    "nvfp4_sparse_attention_shapes",
    "nvfp4_sparse_decode",
    "nvfp4_sparse_decode_blocker",
    "nvfp4_sparse_layer",
    "nvfp4_sparse_layer_blocker",
    "nvfp4_sparse_prefill",
    "nvfp4_sparse_prefill_blocker",
    "prepare_nvfp4_sparse_layer",
]
