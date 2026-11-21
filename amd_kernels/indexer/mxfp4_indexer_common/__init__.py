# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DSA indexer's MXFP4 ops, one entry each, routed by shape to the indexer shape packages: the decode paged
scorer + top-k (``paged``) and the prefill scorer and top-k (``prefill``)."""

from .device import available
from .paged import (
    mxfp4_paged_score_topk,
    mxfp4_paged_score_topk_kernel,
    mxfp4_paged_score_topk_shapes,
    mxfp4_paged_score_topk_unsupported_reason,
    paged_score_topk_preallocate,
    paged_score_topk_prefer,
    paged_score_topk_reset_counts,
)
from .prefill import (
    mxfp4_prefill_score,
    mxfp4_prefill_score_shapes,
    mxfp4_prefill_score_unsupported_reason,
    mxfp4_prefill_topk,
    mxfp4_prefill_topk_shapes,
    mxfp4_prefill_topk_unsupported_reason,
    prefill_topk_workspace_bytes,
)

__all__ = [
    "available",
    "mxfp4_paged_score_topk",
    "mxfp4_paged_score_topk_kernel",
    "mxfp4_paged_score_topk_shapes",
    "mxfp4_paged_score_topk_unsupported_reason",
    "mxfp4_prefill_score",
    "mxfp4_prefill_score_shapes",
    "mxfp4_prefill_score_unsupported_reason",
    "mxfp4_prefill_topk",
    "mxfp4_prefill_topk_shapes",
    "mxfp4_prefill_topk_unsupported_reason",
    "paged_score_topk_preallocate",
    "paged_score_topk_prefer",
    "paged_score_topk_reset_counts",
    "prefill_topk_workspace_bytes",
]
