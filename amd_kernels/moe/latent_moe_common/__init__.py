# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shape-agnostic entries of the latent MoE decode ops (router, routed experts, latent tail, layer tail): each reads
its key from the tensors (or the communicator / runner) and calls the shape package that serves it (``_ROUTES`` of
each module), imported on first use; ``<op>_blocker`` says why a call has no kernel."""
from .dispatch import available, swap_stream_workspaces
from .experts import (
    moe_sort,
    moe_sort_capacity,
    moe_sort_from_aux,
    moe_sort_from_aux_blocker,
    moe_sort_from_aux_unsupported_reason,
    moe_weights,
    moe_weights_blocker,
    moe_weights_unsupported_reason,
    mxfp4_moe_experts,
    mxfp4_moe_experts_blocker,
    mxfp4_moe_experts_shapes,
    mxfp4_moe_experts_unsupported_reason,
    prepare_moe_weights,
    quantize_moe_activations,
    quantize_moe_activations_blocker,
    quantize_moe_activations_unsupported_reason,
)
from .latent_tail import (
    allreduce_latent_rmsnorm_up_proj,
    allreduce_latent_rmsnorm_up_proj_blocker,
    allreduce_latent_rmsnorm_up_proj_shapes,
    allreduce_latent_rmsnorm_up_proj_unsupported_reason,
    allreduce_slots_rmsnorm_up_proj,
    allreduce_slots_rmsnorm_up_proj_blocker,
    allreduce_slots_rmsnorm_up_proj_shapes,
    allreduce_slots_rmsnorm_up_proj_unsupported_reason,
    latent_tail_comm,
    latent_tail_comm_blocker,
    latent_tail_comm_unsupported_reason,
    prepare_latent_tail_comm,
)
from .layer_tail import (
    layer_tail_launch,
    layer_tail_launch_blocker,
    layer_tail_launch_shapes,
    layer_tail_launch_unsupported_reason,
    layer_tail_runner,
    layer_tail_runner_blocker,
    layer_tail_runner_unsupported_reason,
    prepare_layer_tail_runner,
)
from .router import (
    router_aux,
    router_sigmoid_bias_topk,
    router_sigmoid_bias_topk_blocker,
    router_sigmoid_bias_topk_shapes,
    router_sigmoid_bias_topk_unsupported_reason,
    router_sigmoid_bias_topk_weight_blocker,
    router_sigmoid_bias_topk_weight_reason,
)

__all__ = [
    "allreduce_latent_rmsnorm_up_proj", "allreduce_latent_rmsnorm_up_proj_blocker",
    "allreduce_latent_rmsnorm_up_proj_shapes", "allreduce_slots_rmsnorm_up_proj",
    "allreduce_slots_rmsnorm_up_proj_blocker", "allreduce_slots_rmsnorm_up_proj_shapes", "available",
    "latent_tail_comm", "latent_tail_comm_blocker", "layer_tail_launch", "layer_tail_launch_blocker",
    "layer_tail_launch_shapes", "layer_tail_runner", "layer_tail_runner_blocker", "moe_sort", "moe_sort_capacity",
    "moe_sort_from_aux", "moe_sort_from_aux_blocker", "moe_weights", "moe_weights_blocker", "mxfp4_moe_experts",
    "mxfp4_moe_experts_blocker", "mxfp4_moe_experts_shapes", "prepare_latent_tail_comm", "prepare_layer_tail_runner",
    "prepare_moe_weights", "quantize_moe_activations", "quantize_moe_activations_blocker", "router_aux",
    "router_sigmoid_bias_topk", "router_sigmoid_bias_topk_blocker", "router_sigmoid_bias_topk_shapes",
    "router_sigmoid_bias_topk_weight_blocker", "swap_stream_workspaces",
]
