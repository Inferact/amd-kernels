# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 x MXFP4 routed-expert entries: one rank's experts (``host._ROUTES``) and the fused expert-parallel MoE
(``host._FUSED_EP``)."""

from .host import (expert_parallel, expert_parallel_blocker, expert_parallel_shapes, fused_ep_expert_weights,
                   mxfp4_expert_weights, mxfp4_experts, mxfp4_experts_blocker, mxfp4_experts_shapes,
                   prepare_expert_parallel, prepare_mxfp4_expert_weights)

__all__ = ["expert_parallel", "expert_parallel_blocker", "expert_parallel_shapes", "fused_ep_expert_weights",
           "mxfp4_expert_weights", "mxfp4_experts", "mxfp4_experts_blocker", "mxfp4_experts_shapes",
           "prepare_expert_parallel", "prepare_mxfp4_expert_weights"]
