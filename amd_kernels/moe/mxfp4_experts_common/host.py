# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP8 x MXFP4 routed experts: one rank's experts, routed by (hidden, per-rank intermediate, top_k) (``_ROUTES``),
and the fused expert-parallel MoE (dispatch, both GEMMs and combine in one launch per rank), routed by (ranks,
hidden, intermediate, top_k) (``_FUSED_EP``)."""

import torch

from amd_kernels.routes import Routes, arity, first, prepare, previous_form

_ROUTES = Routes(
    "mxfp4_experts",
    {
        (5120, 2304, 6): "amd_kernels.moe.mxfp4_h5120_i2304_e48_top6",
        (5120, 2304, 3): "amd_kernels.moe.mxfp4_h5120_i2304_e16_top3",
    },
    ("hidden", "inter", "top_k"),
)
_FUSED_EP = Routes("expert_parallel", {(8, 5120, 2304, 6): "amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8"},
                   ("ranks", "hidden", "inter", "top_k"))


def mxfp4_experts_shapes() -> frozenset:
    """(hidden, per-rank intermediate, top_k) keys."""
    return _ROUTES.shapes()


def _experts_static(hidden: int, inter: int, local_experts: int, top_k: int, tokens: int) -> str | None:
    key = (hidden, inter, top_k)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    host = _ROUTES.package(key).host
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


def _weights_key(w):
    return _ROUTES.owner(type(w))


@previous_form(_experts_static)
def mxfp4_experts_blocker(x, x_scales, w, topk_weights, topk_ids, **kwargs) -> str | None:
    """None when a package serves this call of one rank's routed experts (``w`` from ``mxfp4_expert_weights``)."""
    key = _weights_key(w)
    if key is None:
        return f"mxfp4_experts: weights {type(w).__name__} not built by mxfp4_expert_weights"
    return _experts_static(key[0], key[1], w.n_experts, key[2], x.shape[0])


def mxfp4_expert_weights(w13, w13_scale, w2, w2_scale, top_k: int, *, hidden: int | None = None,
                         inter: int | None = None, shuffled: bool = False):  # fmt: skip
    """One layer's local experts in the kernel layouts; once, before graph capture. From the row-major checkpoint
    layouts (w13 [experts, 2 I, H / 2], w2 [experts, H, I / 2] packed FP4 with e8m0 scales), or with ``shuffled`` the
    tensors as they are (no copy): AITER's ``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4`` bytes."""
    key = (hidden or 2 * w13.shape[2], inter or w13.shape[1] // 2, top_k)
    return _expert_weights(_ROUTES.package(key), w13, w13_scale, w2, w2_scale, shuffled)


def prepare_mxfp4_expert_weights(w13, w13_scale, w2, w2_scale, top_k: int, *, hidden: int | None = None,
                                 inter: int | None = None, shuffled: bool = False, tokens: int = 1):  # fmt: skip
    """``mxfp4_expert_weights``, or the reason no package serves these experts for up to ``tokens`` rows."""
    hidden, inter = hidden or 2 * w13.shape[2], inter or w13.shape[1] // 2
    return prepare(_experts_static(hidden, inter, w13.shape[0], top_k, tokens),
                   lambda: mxfp4_expert_weights(w13, w13_scale, w2, w2_scale, top_k, hidden=hidden, inter=inter,
                                                shuffled=shuffled))


def mxfp4_experts(x, x_scales, w, topk_weights, topk_ids, **kwargs):
    """The routed experts' weighted sum of each row; ``w`` from ``mxfp4_expert_weights`` (it names the package);
    as the shape package's ``mxfp4_experts``."""
    key = _weights_key(w)
    if key is None:
        raise ValueError(f"mxfp4_experts: weights {type(w).__name__} not built by mxfp4_expert_weights")
    return _ROUTES.package(key).mxfp4_experts(x, x_scales, w, topk_weights, topk_ids, **kwargs)


def expert_parallel_shapes() -> frozenset:
    """(world, hidden, per-expert intermediate, top_k) keys."""
    return _FUSED_EP.shapes()


def _ep_static(world: int, hidden: int, inter: int, top_k: int, experts_per_rank: int, max_tokens: int) -> str | None:
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


@previous_form(_ep_static, arity(6))
def expert_parallel_blocker(rank: int, world: int, group, device, *, hidden: int, inter: int, top_k: int,
                            experts_per_rank: int = 48, max_tokens: int = 1024) -> str | None:  # fmt: skip
    """None when the fused EP MoE serves this layer on one node for up to ``max_tokens`` rows per rank. A collective:
    decide it the same on every rank (from the global maximum of tokens per rank), or the ranks that run it wait for
    the ones that do not."""
    return _ep_static(world, hidden, inter, top_k, experts_per_rank, max_tokens)


def expert_parallel(rank: int, world: int, group, device, *, hidden: int, inter: int, top_k: int,
                    experts_per_rank: int = 48, max_tokens: int = 1024):  # fmt: skip
    """One rank's fused EP MoE (IPC pool, control words, scratch): a collective, created on every rank before graph
    capture, one per process for every layer. ``ep(x, x_scales, w, topk_weights, topk_ids, out=None,
    rows_hint=None)`` runs a layer; every rank calls every layer (an idle rank with 0 rows), one call in flight per
    device; ``ep.warm_up(w)`` compiles every token tier (a collective)."""
    key = (world, hidden, inter, top_k)
    return _FUSED_EP.package(key).ExpertParallel(rank, world, group, device, epr=experts_per_rank)


def prepare_expert_parallel(rank: int, world: int, group, device, *, hidden: int, inter: int, top_k: int,
                            experts_per_rank: int = 48, max_tokens: int = 1024):  # fmt: skip
    """``expert_parallel``'s object, or the reason the fused EP MoE does not serve this layer (a collective: every rank
    gets the same answer from the same arguments)."""
    kw = dict(hidden=hidden, inter=inter, top_k=top_k, experts_per_rank=experts_per_rank, max_tokens=max_tokens)
    return prepare(expert_parallel_blocker(rank, world, group, device, **kw),
                   lambda: expert_parallel(rank, world, group, device, **kw))


def fused_ep_expert_weights(w13, w13_scale, w2, w2_scale, *, world: int, top_k: int, hidden: int | None = None,
                            inter: int | None = None, shuffled: bool = False):  # fmt: skip
    """One layer's local experts in the fused EP kernel's layouts; the inputs as ``mxfp4_expert_weights`` (with
    ``shuffled``: AITER-shuffled bytes used as they are, no copy)."""
    key = (world, hidden or 2 * w13.shape[2], inter or w13.shape[1] // 2, top_k)
    return _expert_weights(_FUSED_EP.package(key), w13, w13_scale, w2, w2_scale, shuffled)
