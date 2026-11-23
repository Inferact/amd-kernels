# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sqrt-softplus top-k router entry (``host._ROUTES``)."""

from .host import (prepare_router_topk, router_shuffle_gate, router_topk, router_topk_blocker, router_topk_shapes,
                   warmup_router_topk)

__all__ = ["prepare_router_topk", "router_shuffle_gate", "router_topk", "router_topk_blocker", "router_topk_shapes",
           "warmup_router_topk"]
