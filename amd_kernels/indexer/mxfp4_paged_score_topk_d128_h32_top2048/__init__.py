# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP4 paged index scorer fused with the per-row top-2048 of the DSA decode indexer (gfx950, FlyDSL).

``reference.py`` is the contract; ``host.py`` the entry and its launch configuration; ``lists.py`` the chunk role
(scores and per-chunk top lists, the logits never reach memory); ``merge.py`` the row-merge role; ``fused.py`` the
one kernel that runs both (chunk workgroups first, then one merge workgroup a row waiting on ready counts).
"""

from .host import cache_views, mxfp4_paged_score_topk, prefer_fused
from .reference import TOPK

__all__ = ["TOPK", "cache_views", "mxfp4_paged_score_topk", "prefer_fused"]
