# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Both TP norm tails of a decoder layer, all-reduce + residual add + Gemma RMSNorm (+ MXFP8 quant, + the FFN between
them): routes the communicator by (hidden, world), read from the norm weight and the CPU group, and a dense layer's
MLP by (hidden, intermediate per rank), read from its down weight [hidden, I], to the shape package's TP module
(``_ROUTES`` / ``_MLP``, imported on first use). The ops are the communicator's methods (``post_attention``,
``pre_attention``, ``capture``)."""

import torch
import torch.distributed as dist

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_PKG = "amd_kernels.norm.gemma_add_rmsnorm_h6144"
_ROUTES = Routes("decoder_ar_norm", {
    (6144, 4): f"{_PKG}.allreduce_add_rmsnorm",
    (6144, 2): f"{_PKG}.allreduce_add_rmsnorm_ranks2",
}, ("hidden", "world"))
_MLP = Routes("ar_norm_dense_mlp", {
    (6144, 3072): f"{_PKG}.allreduce_add_rmsnorm",
    (6144, 6144): f"{_PKG}.allreduce_add_rmsnorm_ranks2",
}, ("hidden", "intermediate"))
_OP = _ROUTES.op


def decoder_ar_norm_shapes() -> frozenset[tuple[int, int]]:
    """The (hidden, world) with communicators."""
    return _ROUTES.shapes()


def decoder_ar_norm_world_blocker(world_size: int, hidden_size: int) -> str | None:
    """None when this process can build the communicators of ``world_size`` ranks at ``hidden_size`` (a load-time
    check before the group exists)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = (hidden_size, world_size)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    reason = _ROUTES.package(key).unsupported_reason(world_size, hidden_size)
    return None if reason is None else f"{_OP}: {reason}"


def decoder_ar_norm(rank: int, cpu_group, device, weight, eps: float, *, moe=None):
    """The routed module's ``DecoderArNorm`` (collective on the CPU group, warmed up; with ``moe``, one layer's
    ``mxfp4_moe_weights``, also the MoE modes of ``post_attention``)."""
    key = (weight.numel(), dist.get_world_size(cpu_group))
    return _ROUTES.package(key).DecoderArNorm(rank, cpu_group, device, weight, eps, moe=moe)


def decoder_ar_norm_blocker(rank: int, cpu_group, device, weight, eps: float, *, moe=None) -> str | None:
    """None when ``decoder_ar_norm`` with these arguments builds the communicators, else why not."""
    world = dist.get_world_size(cpu_group)
    reason = decoder_ar_norm_world_blocker(world, weight.numel())
    if reason is not None:
        return reason
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        return f"{_OP}: norm weight {weight.dtype} (contiguous bf16)"
    if dist.get_rank(cpu_group) != rank:
        return f"{_OP}: rank {rank} is not this process's rank {dist.get_rank(cpu_group)} in the group"
    return None


def prepare_decoder_ar_norm(rank: int, cpu_group, device, weight, eps: float, *, moe=None):
    """``decoder_ar_norm(...)``, or the reason it cannot be built."""
    return prepare(decoder_ar_norm_blocker(rank, cpu_group, device, weight, eps, moe=moe),
                   lambda: decoder_ar_norm(rank, cpu_group, device, weight, eps, moe=moe))


def _mlp_key(down):
    return tuple(down.shape) if down.dim() == 2 else None


def ar_norm_dense_mlp(gate_up, gate_up_scale, down, down_scale, alpha: float, beta: float, limit: float):
    """A dense layer's MLP for ``post_attention`` (preshuffled MXFP8 gate_up / down with their scales, SwiGLU-OAI's
    alpha, beta, limit): the routed module's ``DenseMlp``."""
    return _MLP.package(_mlp_key(down)).DenseMlp(gate_up, gate_up_scale, down, down_scale, alpha, beta, limit)


def ar_norm_dense_mlp_blocker(gate_up, gate_up_scale, down, down_scale, alpha: float, beta: float,
                              limit: float) -> str | None:
    """None when ``ar_norm_dense_mlp`` takes these weights, else why not."""
    op = _MLP.op
    key = _mlp_key(down)
    if key is None:
        return f"{op}: down {tuple(down.shape)} ([hidden, I])"
    reason = _MLP.blocker(key)
    if reason is not None:
        return reason
    if not _MLP.package(key).dense_mlp_supported(gate_up, down):
        return f"{op}: gate_up {tuple(gate_up.shape)} {gate_up.dtype}, down {tuple(down.shape)} {down.dtype}"
    return None


def prepare_ar_norm_dense_mlp(gate_up, gate_up_scale, down, down_scale, alpha: float, beta: float, limit: float):
    """``ar_norm_dense_mlp(...)``, or the reason it cannot be built."""
    args = (gate_up, gate_up_scale, down, down_scale, alpha, beta, limit)
    return prepare(ar_norm_dense_mlp_blocker(*args), lambda: ar_norm_dense_mlp(*args))
