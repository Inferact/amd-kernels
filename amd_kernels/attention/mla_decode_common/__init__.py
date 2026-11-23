# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSA / MLA decode attention: one entry per op, routed by the shape read from the operands to the shape package
(``_ROUTES`` of each module); a shape package is imported on the first call of its shape. An uncovered call gets a
reason from ``*_blocker``."""

from .combine import (
    lse_combine_comm,
    lse_combine_comm_blocker,
    lse_combine_comm_unsupported_reason,
    lse_combine_vup,
    lse_combine_vup_blocker,
    lse_combine_vup_kernel,
    lse_combine_vup_shapes,
    lse_combine_vup_unsupported_reason,
    prepare_lse_combine_comm,
    prepare_w_uv,
)
from .dense import (
    fp8_dense_mla_decode,
    fp8_dense_mla_decode_blocker,
    fp8_dense_mla_decode_shapes,
    fp8_dense_mla_decode_unsupported_reason,
)
from .prologue import (
    indexer_prep,
    indexer_prep_blocker,
    indexer_prep_kernel,
    indexer_prep_shapes,
    indexer_prep_unsupported_reason,
    qkv_a_norm_kv_insert,
    qkv_a_norm_kv_insert_blocker,
    qkv_a_norm_kv_insert_kernel,
    qkv_a_norm_kv_insert_shapes,
    qkv_a_norm_kv_insert_unsupported_reason,
)
from .query import (
    pack_q_b,
    pack_w_uk,
    prepare_q_gather_comm,
    q_bproj_wuk_gather,
    q_bproj_wuk_gather_blocker,
    q_bproj_wuk_gather_kernel,
    q_bproj_wuk_gather_shapes,
    q_bproj_wuk_gather_unsupported_reason,
    q_gather_comm,
    q_gather_comm_blocker,
    q_gather_comm_unsupported_reason,
)
from .routing import available
from .sparse import (
    fp8_sparse_mla_decode,
    fp8_sparse_mla_decode_blocker,
    fp8_sparse_mla_decode_kernel,
    fp8_sparse_mla_decode_shapes,
    fp8_sparse_mla_decode_unsupported_reason,
)

__all__ = [
    "available",
    "fp8_dense_mla_decode",
    "fp8_dense_mla_decode_blocker",
    "fp8_dense_mla_decode_shapes",
    "fp8_sparse_mla_decode",
    "fp8_sparse_mla_decode_blocker",
    "fp8_sparse_mla_decode_kernel",
    "fp8_sparse_mla_decode_shapes",
    "indexer_prep",
    "indexer_prep_blocker",
    "indexer_prep_kernel",
    "indexer_prep_shapes",
    "lse_combine_comm",
    "lse_combine_comm_blocker",
    "lse_combine_vup",
    "lse_combine_vup_blocker",
    "lse_combine_vup_kernel",
    "lse_combine_vup_shapes",
    "pack_q_b",
    "pack_w_uk",
    "prepare_lse_combine_comm",
    "prepare_q_gather_comm",
    "prepare_w_uv",
    "q_bproj_wuk_gather",
    "q_bproj_wuk_gather_blocker",
    "q_bproj_wuk_gather_kernel",
    "q_bproj_wuk_gather_shapes",
    "q_gather_comm",
    "q_gather_comm_blocker",
    "qkv_a_norm_kv_insert",
    "qkv_a_norm_kv_insert_blocker",
    "qkv_a_norm_kv_insert_kernel",
    "qkv_a_norm_kv_insert_shapes",
]
