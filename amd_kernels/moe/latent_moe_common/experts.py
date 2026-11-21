# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the routed MXFP4 experts (A4W4, slot sum) and their weights, sort and activation quantizer, routed by the
shape of the gate/up weight ``w13`` [E, 2 I, H / 2]."""
import functools
import importlib

import torch

from . import dispatch

_SHAPE_HOSTS = {(896, 768, 1792): "amd_kernels.moe.mxfp4_h3584_i3072_e896_top16.host"}  # w13 [E, 2 I, H / 2]
_NAMES = ("E", "2I", "H/2")


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


def _host(w13, op):
    key = tuple(w13.shape)
    host = _shape_host(key)
    if host is None:
        raise ValueError(dispatch.no_kernel(op, _NAMES, key, _SHAPE_HOSTS))
    return host


def _package(host) -> str:
    return host.__name__.split(".")[-2]


def _routed(op, weights):
    """(routed host, None) for ``weights`` of a routed package, else (None, reason)."""
    if not dispatch.available():
        return None, f"{op}: needs gfx950 with FlyDSL"
    w13 = getattr(weights, "w13", None)
    key = tuple(w13.shape) if isinstance(w13, torch.Tensor) else ()
    host = _shape_host(key)
    if host is None:
        return None, dispatch.no_kernel(op, _NAMES, key, _SHAPE_HOSTS)
    if type(weights) is not host.MoeWeights:
        return None, f"{op}: weights not built by moe_weights"
    return host, None


def moe_weights(w13, s13, w2, s2):
    """Views of the routed experts' weights in the layout vLLM / AITER hold (preshuffled MXFP4 codes, E8M0 scales)."""
    return _host(w13, "moe_weights").MoeWeights.from_shuffled(w13, s13, w2, s2)


def moe_weights_unsupported_reason(w13, s13, w2, s2) -> str | None:
    op = "moe_weights"
    if not dispatch.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = tuple(w13.shape)
    host = _shape_host(key)
    if host is None:
        return dispatch.no_kernel(op, _NAMES, key, _SHAPE_HOSTS)
    if any(t.element_size() != 1 for t in (w13, s13, w2, s2)):
        return f"{op}: operand layout ({_package(host)}.MoeWeights.from_shuffled)"
    try:
        host.MoeWeights.from_shuffled(w13, s13, w2, s2)
    except (AssertionError, RuntimeError):
        return f"{op}: operand layout ({_package(host)}.MoeWeights.from_shuffled)"
    return None


def moe_sort_from_aux(aux, *, weights):
    """The experts' sort from the router's ``aux`` (``router_aux``)."""
    return _host(weights.w13, "moe_sort_from_aux").Sort.from_aux(aux)


def moe_sort_from_aux_unsupported_reason(aux, *, weights) -> str | None:
    """None when ``aux`` is a sort the experts of ``weights`` take (top-k, expert count and width match)."""
    op = "moe_sort_from_aux"
    host, reason = _routed(op, weights)
    if reason is not None:
        return reason
    keys = ("sorted_token_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids", "reverse_sorted")
    if any(not isinstance(aux.get(k), torch.Tensor) for k in keys):
        return f"{op}: aux lacks the sort tensors"
    if any(aux[k].device != weights.w13.device for k in keys):
        return f"{op}: aux is not on the weights' device"
    routes = aux["reverse_sorted"].numel()
    mb = aux.get("moe_buf")
    m = mb.shape[0] if mb is not None else routes // host.TOPK
    cap = host.sort_capacity(m)
    if (routes != m * host.TOPK or (mb is not None and tuple(mb.shape) != (m, host.H))
            or aux["sorted_token_ids"].numel() < cap or aux["sorted_expert_ids"].numel() < cap // host.BM):
        rows = aux["sorted_token_ids"].numel()
        return (f"{op}: aux of {routes} routes / {rows} sorted rows is not a top-{host.TOPK} sort over {host.E} "
                f"experts of width {host.H}")
    return None


def moe_sort(topk_ids, topk_weights, *, weights):
    """The experts' sort of (topk_ids, topk_weights) by AITER's ``moe_sorting`` (the router's own sort is faster)."""
    return _host(weights.w13, "moe_sort").moe_sort(topk_ids, topk_weights)


def moe_sort_capacity(m: int, *, weights) -> int:
    """Rows of the sort of ``m`` tokens (the slot buffer rows ``reduce=False`` hands over)."""
    return _host(weights.w13, "moe_sort_capacity").sort_capacity(m)


def quantize_moe_activations(x, codes=None, scales=None, *, weights):
    """(codes u8 [m, H / 2], E8M0 scales u8 [m, H / 32]) of bf16 ``x`` [m, H], the experts' activation format."""
    return _host(weights.w13, "quantize_moe_activations").quantize_activations(x, codes, scales)


def quantize_moe_activations_unsupported_reason(x, codes=None, scales=None, *, weights) -> str | None:
    op = "quantize_moe_activations"
    host, reason = _routed(op, weights)
    if reason is not None:
        return reason
    h = host.H
    if (x.ndim != 2 or x.shape[1] != h or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.device != weights.w13.device):
        return f"{op}: x must be contiguous bf16 [M, {h}] on the weights' device"
    m = x.shape[0]
    if (codes is None) != (scales is None):
        return f"{op}: codes and scales are given together or not at all"
    if codes is not None and any(
            t.element_size() != 1 or not t.is_contiguous() or t.numel() < m * n or t.device != x.device
            for t, n in ((codes, h // 2), (scales, h // 32))):
        return f"{op}: codes / scales must be contiguous 1-byte [{m}, {h // 2}] / [{m}, {h // 32}] on x's device"
    return None


def mxfp4_moe_experts(x_codes, x_scales, weights, sort, out=None, *, fused=True, reduce=True):
    """The routed experts of ``m`` tokens: bf16 [m, H] (slot sum in slot order), or with ``reduce=False`` the slot
    buffer whose row ``sort.reverse_sorted[16 t + k]`` is token t's slot k."""
    return _host(weights.w13, "mxfp4_moe_experts").mxfp4_moe_experts(x_codes, x_scales, weights, sort, out, fused=fused,
                                                                     reduce=reduce)


def mxfp4_moe_experts_shapes() -> frozenset:
    return frozenset(_SHAPE_HOSTS)


def mxfp4_moe_experts_unsupported_reason(x_codes, x_scales, weights, sort, out=None, *, fused=True,
                                         reduce=True) -> str | None:
    op = "mxfp4_moe_experts"
    host, reason = _routed(op, weights)
    if reason is not None:
        return reason
    if type(sort) is not host.Sort:
        return f"{op}: sort not built by moe_sort_from_aux / moe_sort"
    h, dev = host.H, weights.w13.device
    m = x_codes.shape[0] if x_codes.ndim == 2 else 0
    if not 1 <= m <= host.MAX_M:
        return f"{op}: {m} rows (the kernel takes 1..{host.MAX_M})"
    for t, n in ((x_codes, h // 2), (x_scales, h // 32)):
        if t.element_size() != 1 or tuple(t.shape) != (m, n) or not t.is_contiguous() or t.device != dev:
            return f"{op}: x_codes / x_scales must be contiguous 1-byte [{m}, {h // 2}] / [{m}, {h // 32}]"
    cap = host.sort_capacity(m)
    if (sort.sorted_token_ids.numel() < cap or sort.sorted_expert_ids.numel() < cap // host.BM
            or sort.reverse_sorted.numel() < m * host.TOPK):
        return f"{op}: sort is smaller than the sort of {m} tokens"
    if reduce and out is not None and (tuple(out.shape) != (m, h) or out.dtype != torch.bfloat16
                                       or not out.is_contiguous() or out.device != dev):
        return f"{op}: out must be contiguous bf16 [{m}, {h}]"
    return None
