# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic entries of the latent MoE decode ops (router, routed experts, latent tail, layer tail): each reads
its key from the tensors (or the communicator / runner) and calls the shape package that serves it, imported on first
use; ``<op>_unsupported_reason`` says why a call has no kernel."""
from .dispatch import available
from .experts import (
    moe_sort,
    moe_sort_capacity,
    moe_sort_from_aux,
    moe_sort_from_aux_unsupported_reason,
    moe_weights,
    moe_weights_unsupported_reason,
    mxfp4_moe_experts,
    mxfp4_moe_experts_shapes,
    mxfp4_moe_experts_unsupported_reason,
    quantize_moe_activations,
    quantize_moe_activations_unsupported_reason,
)
from .latent_tail import (
    allreduce_latent_rmsnorm_up_proj,
    allreduce_latent_rmsnorm_up_proj_shapes,
    allreduce_latent_rmsnorm_up_proj_unsupported_reason,
    allreduce_slots_rmsnorm_up_proj,
    allreduce_slots_rmsnorm_up_proj_shapes,
    allreduce_slots_rmsnorm_up_proj_unsupported_reason,
    latent_tail_comm,
    latent_tail_comm_unsupported_reason,
)
from .layer_tail import (
    layer_tail_launch,
    layer_tail_launch_shapes,
    layer_tail_launch_unsupported_reason,
    layer_tail_runner,
    layer_tail_runner_unsupported_reason,
)
from .router import (
    router_aux,
    router_sigmoid_bias_topk,
    router_sigmoid_bias_topk_shapes,
    router_sigmoid_bias_topk_unsupported_reason,
    router_sigmoid_bias_topk_weight_reason,
)

__all__ = [
    "allreduce_latent_rmsnorm_up_proj", "allreduce_latent_rmsnorm_up_proj_shapes",
    "allreduce_latent_rmsnorm_up_proj_unsupported_reason", "allreduce_slots_rmsnorm_up_proj",
    "allreduce_slots_rmsnorm_up_proj_shapes", "allreduce_slots_rmsnorm_up_proj_unsupported_reason", "available",
    "latent_tail_comm", "latent_tail_comm_unsupported_reason", "layer_tail_launch", "layer_tail_launch_shapes",
    "layer_tail_launch_unsupported_reason", "layer_tail_runner", "layer_tail_runner_unsupported_reason", "moe_sort",
    "moe_sort_capacity", "moe_sort_from_aux", "moe_sort_from_aux_unsupported_reason", "moe_weights",
    "moe_weights_unsupported_reason", "mxfp4_moe_experts", "mxfp4_moe_experts_shapes",
    "mxfp4_moe_experts_unsupported_reason", "quantize_moe_activations", "quantize_moe_activations_unsupported_reason",
    "router_aux", "router_sigmoid_bias_topk", "router_sigmoid_bias_topk_shapes",
    "router_sigmoid_bias_topk_unsupported_reason", "router_sigmoid_bias_topk_weight_reason",
]
