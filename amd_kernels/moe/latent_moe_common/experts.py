# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the routed MXFP4 experts (A4W4, slot sum) and their weights, sort and activation quantizer, routed by the
shape of the gate/up weight ``w13`` [E, 2 I, H / 2]."""
import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare, renamed

from . import dispatch

_NAMES = ("E", "2I", "H/2")  # w13 [E, 2 I, H / 2]
_ROUTES = Routes("mxfp4_moe_experts", {(896, 768, 1792): "amd_kernels.moe.mxfp4_h3584_i3072_e896_top16.host"}, _NAMES)
_WEIGHTS = Routes("moe_weights", _ROUTES.hosts, _NAMES)
_SORT = Routes("moe_sort", _ROUTES.hosts, _NAMES)
_SORT_FROM_AUX = Routes("moe_sort_from_aux", _ROUTES.hosts, _NAMES)
_SORT_CAPACITY = Routes("moe_sort_capacity", _ROUTES.hosts, _NAMES)
_QUANTIZE = Routes("quantize_moe_activations", _ROUTES.hosts, _NAMES)


def _package_name(host) -> str:
    return host.__name__.split(".")[-2]


def _routed(op_routes, weights):
    """(routed host, None) for ``weights`` of a routed package, else (None, reason)."""
    op = op_routes.op
    if not routes.available():
        return None, f"{op}: needs gfx950 with FlyDSL"
    w13 = getattr(weights, "w13", None)
    if not isinstance(w13, torch.Tensor):
        return None, f"{op}: weights not built by moe_weights"
    key = tuple(w13.shape)
    reason = dispatch.blocker(op_routes, key)
    if reason is not None:
        return None, reason
    host = op_routes.package(key)
    if type(weights) is not host.MoeWeights:
        return None, f"{op}: weights not built by moe_weights"
    return host, None


def moe_weights(w13, s13, w2, s2):
    """Views of the routed experts' weights in the layout vLLM / AITER hold (preshuffled MXFP4 codes, E8M0 scales)."""
    return _WEIGHTS.package(tuple(w13.shape)).MoeWeights.from_shuffled(w13, s13, w2, s2)


def moe_weights_blocker(w13, s13, w2, s2) -> str | None:
    op = _WEIGHTS.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = tuple(w13.shape)
    reason = dispatch.blocker(_WEIGHTS, key)
    if reason is not None:
        return reason
    host = _WEIGHTS.package(key)
    if any(t.element_size() != 1 for t in (w13, s13, w2, s2)):
        return f"{op}: operand layout ({_package_name(host)}.MoeWeights.from_shuffled)"
    try:
        host.MoeWeights.from_shuffled(w13, s13, w2, s2)
    except (AssertionError, RuntimeError):
        return f"{op}: operand layout ({_package_name(host)}.MoeWeights.from_shuffled)"
    return None


def prepare_moe_weights(w13, s13, w2, s2):
    """``moe_weights(...)``, or the reason the experts cannot take these weights."""
    return prepare(moe_weights_blocker(w13, s13, w2, s2), lambda: moe_weights(w13, s13, w2, s2))


def moe_sort_from_aux(aux, *, weights):
    """The experts' sort from the router's ``aux`` (``router_aux``)."""
    return _SORT_FROM_AUX.package(tuple(weights.w13.shape)).Sort.from_aux(aux)


def moe_sort_from_aux_blocker(aux, *, weights) -> str | None:
    """None when ``aux`` is a sort the experts of ``weights`` take (top-k, expert count and width match)."""
    op = _SORT_FROM_AUX.op
    host, reason = _routed(_SORT_FROM_AUX, weights)
    if reason is not None:
        return reason
    keys = ("sorted_token_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids", "reverse_sorted")
    if any(not isinstance(aux.get(k), torch.Tensor) for k in keys):
        return f"{op}: aux lacks the sort tensors"
    if any(aux[k].device != weights.w13.device for k in keys):
        return f"{op}: aux is not on the weights' device"
    n_routes = aux["reverse_sorted"].numel()
    mb = aux.get("moe_buf")
    m = mb.shape[0] if mb is not None else n_routes // host.TOPK
    cap = host.sort_capacity(m)
    if (n_routes != m * host.TOPK or (mb is not None and tuple(mb.shape) != (m, host.H))
            or aux["sorted_token_ids"].numel() < cap or aux["sorted_expert_ids"].numel() < cap // host.BM):
        rows = aux["sorted_token_ids"].numel()
        return (f"{op}: aux of {n_routes} routes / {rows} sorted rows is not a top-{host.TOPK} sort over {host.E} "
                f"experts of width {host.H}")
    return None


def moe_sort(topk_ids, topk_weights, *, weights):
    """The experts' sort of (topk_ids, topk_weights) by AITER's ``moe_sorting`` (the router's own sort is faster)."""
    return _SORT.package(tuple(weights.w13.shape)).moe_sort(topk_ids, topk_weights)


def moe_sort_capacity(m: int, *, weights) -> int:
    """Rows of the sort of ``m`` tokens (the slot buffer rows ``reduce=False`` hands over)."""
    return _SORT_CAPACITY.package(tuple(weights.w13.shape)).sort_capacity(m)


def quantize_moe_activations(x, codes=None, scales=None, *, weights):
    """(codes u8 [m, H / 2], E8M0 scales u8 [m, H / 32]) of bf16 ``x`` [m, H], the experts' activation format."""
    return _QUANTIZE.package(tuple(weights.w13.shape)).quantize_activations(x, codes, scales)


def quantize_moe_activations_blocker(x, codes=None, scales=None, *, weights) -> str | None:
    op = _QUANTIZE.op
    host, reason = _routed(_QUANTIZE, weights)
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
    return _ROUTES.package(tuple(weights.w13.shape)).mxfp4_moe_experts(x_codes, x_scales, weights, sort, out,
                                                                       fused=fused, reduce=reduce)


def mxfp4_moe_experts_shapes() -> frozenset:
    return _ROUTES.shapes()


def mxfp4_moe_experts_blocker(x_codes, x_scales, weights, sort, out=None, *, fused=True,
                              reduce=True) -> str | None:
    op = _ROUTES.op
    host, reason = _routed(_ROUTES, weights)
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


moe_weights_unsupported_reason = renamed(moe_weights_blocker, "moe_weights_unsupported_reason")
moe_sort_from_aux_unsupported_reason = renamed(moe_sort_from_aux_blocker, "moe_sort_from_aux_unsupported_reason")
quantize_moe_activations_unsupported_reason = renamed(quantize_moe_activations_blocker,
                                                      "quantize_moe_activations_unsupported_reason")
mxfp4_moe_experts_unsupported_reason = renamed(mxfp4_moe_experts_blocker, "mxfp4_moe_experts_unsupported_reason")
