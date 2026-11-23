# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE op entries (``amd_kernels.routes``): the sqrt-softplus router, one rank's routed experts, fused EP."""

import torch

from amd_kernels.routes import Routes, check, first

_ROUTER = Routes(
    "router_topk",
    {
        (5120, 384, 6): "amd_kernels.moe.router_sqrtsoftplus_top6_h5120_e384",
        (5120, 128, 3): "amd_kernels.moe.router_sqrtsoftplus_top3_h5120_e128",
    },
)
_EXPERTS = Routes(
    "mxfp4_experts",
    {
        (5120, 2304, 6): "amd_kernels.moe.mxfp4_h5120_i2304_e48_top6",
        (5120, 2304, 3): "amd_kernels.moe.mxfp4_h5120_i2304_e16_top3",
    },
)
_FUSED_EP = Routes(
    "expert_parallel", {(8, 5120, 2304, 6): "amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8"}
)


def router_topk_shapes() -> frozenset:
    """(hidden, experts, top_k) keys."""
    return _ROUTER.shapes()


def router_topk_blocker(x, bias, top_k: int) -> str | None:
    """None when a package serves the gate GEMM + biased sqrt-softplus top-``top_k`` of ``x`` [rows, hidden] over
    ``bias`` [experts] experts."""
    if x.dim() != 2 or bias.dim() != 1:
        return "router_topk: x must be [rows, hidden], bias [experts]"
    return _ROUTER.blocker((x.shape[1], bias.shape[0], top_k))


def router_topk(x, weight, bias, top_k: int, routed_scaling_factor: float = 1.5, **kwargs):
    """``(topk_weights, topk_ids)`` of each row; keyword arguments (``is_padding``, output buffers,
    ``weight_shuffled``) as the shape package's ``router_topk``."""
    check(router_topk_blocker(x, bias, top_k))
    return _ROUTER.package((x.shape[1], bias.shape[0], top_k)).router_topk(x, weight, bias, routed_scaling_factor,
                                                                            **kwargs)


def router_shuffle_gate(weight, top_k: int):
    """The gate ``weight`` [experts, hidden] in the load-time layout of ``router_topk(..., weight_shuffled=True)``."""
    return _ROUTER.package((weight.shape[1], weight.shape[0], top_k)).shuffle_gate(weight)


def warmup_router_topk(hidden: int, experts: int, top_k: int, device=None) -> None:
    """Compiles every row-count bucket of the router (before graph capture)."""
    _ROUTER.package((hidden, experts, top_k)).warmup_router_topk(device)


def mxfp4_experts_shapes() -> frozenset:
    """(hidden, per-rank intermediate, top_k) keys."""
    return _EXPERTS.shapes()


def mxfp4_experts_blocker(hidden: int, inter: int, local_experts: int, top_k: int, tokens: int) -> str | None:
    """None when a package serves one rank's MXFP8 x MXFP4 routed experts for this layer and up to ``tokens`` rows."""
    key = (hidden, inter, top_k)
    reason = _EXPERTS.blocker(key)
    if reason is not None:
        return reason
    host = _EXPERTS.package(key).host
    return first(
        None if 1 <= local_experts <= host.NE_MAX else f"mxfp4_experts: {local_experts} local experts outside "
        f"1..{host.NE_MAX}",
        None if tokens <= host.MAX_TOKENS else f"mxfp4_experts: {tokens} tokens above {host.MAX_TOKENS}",
    )


def _expert_weights(pkg, w13, w13_scale, w2, w2_scale, shuffled: bool):
    if not shuffled:
        return pkg.ExpertWeights.from_row_major(w13, w13_scale, w2, w2_scale)
    u8 = torch.uint8
    return pkg.ExpertWeights(w13.view(u8), w13_scale.view(u8), w2.view(u8), w2_scale.view(u8))


def mxfp4_expert_weights(w13, w13_scale, w2, w2_scale, top_k: int, *, hidden: int | None = None,
                         inter: int | None = None, shuffled: bool = False):  # fmt: skip
    """One layer's local experts in the kernel layouts; once, before graph capture. From the row-major checkpoint
    layouts (w13 [experts, 2 I, H / 2], w2 [experts, H, I / 2] packed FP4 with e8m0 scales), or with ``shuffled`` the
    tensors as they are (no copy): AITER's ``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4`` bytes, e.g. MegaMoE's
    weights shuffled in place. ``hidden`` / ``inter`` default to the row-major shapes."""
    key = (hidden or 2 * w13.shape[2], inter or w13.shape[1] // 2, top_k)
    return _expert_weights(_EXPERTS.package(key), w13, w13_scale, w2, w2_scale, shuffled)


def _experts_package(w):
    name = type(w).__module__.rsplit(".", 1)[0]
    if name not in _EXPERTS.hosts.values():
        raise ValueError(f"mxfp4_experts: weights of {name}, not of an experts package (mxfp4_expert_weights)")
    return _EXPERTS.package(next(k for k, v in _EXPERTS.hosts.items() if v == name))


def mxfp4_experts(x, x_scales, w, topk_weights, topk_ids, **kwargs):
    """The routed experts' weighted sum of each row; ``w`` from ``mxfp4_expert_weights`` (it names the package);
    as the shape package's ``mxfp4_experts``."""
    return _experts_package(w).mxfp4_experts(x, x_scales, w, topk_weights, topk_ids, **kwargs)


def expert_parallel_shapes() -> frozenset:
    """(world, hidden, per-expert intermediate, top_k) keys."""
    return _FUSED_EP.shapes()


def expert_parallel_blocker(world: int, hidden: int, inter: int, top_k: int, experts_per_rank: int,
                            max_tokens: int) -> str | None:  # fmt: skip
    """None when the fused expert-parallel MoE serves this layer on one node. A collective: decide it the same on every
    rank (from the global maximum of tokens per rank), or the ranks that run it wait for the ones that do not."""
    key = (world, hidden, inter, top_k)
    reason = _FUSED_EP.blocker(key)
    if reason is not None:
        return reason
    host = _FUSED_EP.package(key).host
    return first(
        None if 1 <= experts_per_rank <= host.NE_MAX else f"expert_parallel: {experts_per_rank} experts per rank "
        f"outside 1..{host.NE_MAX}",
        None if max_tokens <= host.MAX_TOKENS else f"expert_parallel: {max_tokens} tokens per rank above "
        f"{host.MAX_TOKENS}",
    )


def expert_parallel(rank: int, world: int, group, device, *, hidden: int, inter: int, top_k: int,
                    experts_per_rank: int = 48):  # fmt: skip
    """One rank's fused EP MoE (IPC pool, control words, scratch): a collective, created on every rank before graph
    capture, one per process for every layer. ``ep(x, x_scales, w, topk_weights, topk_ids, out=None,
    rows_hint=None)`` runs a layer; every rank calls every layer (an idle rank with 0 rows), one call in flight per
    device; ``ep.warm_up(w)`` compiles every token tier (a collective)."""
    key = (world, hidden, inter, top_k)
    return _FUSED_EP.package(key).ExpertParallel(rank, world, group, device, epr=experts_per_rank)


def fused_ep_expert_weights(w13, w13_scale, w2, w2_scale, *, world: int, top_k: int, hidden: int | None = None,
                            inter: int | None = None, shuffled: bool = False):  # fmt: skip
    """One layer's local experts in the fused EP kernel's layouts; the inputs as ``mxfp4_expert_weights`` (with
    ``shuffled``: AITER-shuffled bytes used as they are, no copy)."""
    key = (world, hidden or 2 * w13.shape[2], inter or w13.shape[1] // 2, top_k)
    return _expert_weights(_FUSED_EP.package(key), w13, w13_scale, w2, w2_scale, shuffled)
