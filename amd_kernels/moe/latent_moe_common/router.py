# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry of the MoE router (gate GEMM, sigmoid + correction bias, top-k, renormalised, optionally the BM=16 sort the
experts take), routed by (experts, hidden, top-k) of the gate weight."""
import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

from . import dispatch

_ROUTES = Routes("router_sigmoid_bias_topk",
                 {(896, 7168, 16): "amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168.host"}, ("E", "H", "topk"))
_OP = _ROUTES.op
_AUX_KEYS = ("sorted_token_ids", "sorted_weights", "sorted_expert_ids", "num_valid_ids", "m_indices", "reverse_sorted")


def _package_name(host) -> str:
    return host.__name__.split(".")[-2]


def router_sigmoid_bias_topk(x, w, bias, routed_scaling_factor=1.0, out=None, *, topk=16, shuffled=False, aux=None):
    """(topk_weights fp32 [M, topk], topk_ids int32 [M, topk]) of ``x`` bf16 [M, H] under the gate ``w`` bf16 [E, H]
    and the fp32 correction ``bias`` [E]; ``aux`` (``router_aux``) receives the experts' sort."""
    host = _ROUTES.package((*w.shape, topk))
    return getattr(host, _package_name(host))(x, w, bias, routed_scaling_factor, out, shuffled=shuffled, aux=aux)


def router_aux(w, m: int, device, *, topk=16, zero_moe_buf=True) -> dict:
    """The sort tensors of ``m`` tokens the router fills (``aux=``) for ``moe_sort_from_aux``."""
    return _ROUTES.package((*w.shape, topk)).make_aux(m, device, zero_moe_buf)


def router_sigmoid_bias_topk_shapes() -> frozenset:
    return _ROUTES.shapes()


def router_sigmoid_bias_topk_weight_blocker(w, bias=None, *, topk=16) -> str | None:
    """None when a layer's gate weight and bias (load time, no activation yet) have a kernel, else why not."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = (*w.shape, topk)
    reason = dispatch.blocker(_ROUTES, key)
    if reason is not None:
        return reason
    host = _ROUTES.package(key)
    if not host.supports(w, bias):
        return f"{_OP}: operand layout ({_package_name(host)}.supports)"
    return None


def router_sigmoid_bias_topk_blocker(x, w, bias, routed_scaling_factor=1.0, out=None, *, topk=16, shuffled=False,
                                     aux=None) -> str | None:
    reason = router_sigmoid_bias_topk_weight_blocker(w, bias, topk=topk)
    if reason is not None:
        return reason
    if bias is None or not bias.is_contiguous() or bias.device != w.device:
        return f"{_OP}: needs the contiguous fp32 correction bias on the weight's device"
    host = _ROUTES.package((*w.shape, topk))
    h = w.shape[1]
    if x.ndim != 2 or x.shape[1] != h or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != w.device:
        return f"{_OP}: x must be contiguous bf16 [M, {h}] on the weight's device"
    m = x.shape[0]
    if not 1 <= m <= host.MAX_M:
        return f"{_OP}: {m} rows (the kernel takes 1..{host.MAX_M})"
    if out is not None:
        ow, oi = out
        if (ow.shape != (m, topk) or oi.shape != (m, topk) or ow.dtype != torch.float32 or oi.dtype != torch.int32
                or ow.stride() != oi.stride() or ow.stride(1) != 1 or ow.device != x.device or oi.device != x.device):
            return f"{_OP}: out must be (fp32, int32) [{m}, {topk}] with equal strides, unit inner stride"
    if aux is not None:
        cap = host.aux_capacity(m)
        sizes = (cap, cap, cap // 16, 2, cap, m * topk)
        if any(k not in aux for k in _AUX_KEYS) or any(
                not aux[k].is_contiguous() or aux[k].numel() < n or aux[k].device != x.device
                for k, n in zip(_AUX_KEYS, sizes)):
            return f"{_OP}: aux is not router_aux(w, {m}, ...)"
        mb = aux.get("moe_buf")
        if mb is not None and (mb.dtype != torch.bfloat16 or not mb.is_contiguous() or mb.ndim != 2
                               or tuple(mb.shape) != (m, host.MOE_BUF_COLS) or mb.device != x.device):
            return f"{_OP}: aux moe_buf is not router_aux(w, {m}, ...)'s"
    return None


router_sigmoid_bias_topk_unsupported_reason = renamed(router_sigmoid_bias_topk_blocker,
                                                      "router_sigmoid_bias_topk_unsupported_reason")
router_sigmoid_bias_topk_weight_reason = renamed(router_sigmoid_bias_topk_weight_blocker,
                                                 "router_sigmoid_bias_topk_weight_reason")
