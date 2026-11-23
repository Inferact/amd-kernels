# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DCP top-k score gather + merge, routed by shape to the collective shape packages (``host._ROUTES``)."""

from amd_kernels.routes import available

from .host import (
    prepare_topk_gather_merge_comm,
    topk_gather_merge_blocker,
    topk_gather_merge_comm,
    topk_gather_merge_comm_blocker,
    topk_gather_merge_comm_unsupported_reason,
    topk_gather_merge_shapes,
    topk_gather_merge_unsupported_reason,
)

__all__ = [
    "available",
    "prepare_topk_gather_merge_comm",
    "topk_gather_merge_blocker",
    "topk_gather_merge_comm",
    "topk_gather_merge_comm_blocker",
    "topk_gather_merge_shapes",
]
