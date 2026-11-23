# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Latent down projection + MXFP4 quant: routes by the weight's (N, K) to the shape package's host, whose op is the
function named like the package."""
import functools
import sys

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

OP = "bf16_latent_down_mxfp4_quant"
_DIMS = ("n", "k")
_ROUTES = Routes(OP, {
    (3584, 7168): "amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584.host",
}, _DIMS)


@functools.cache
def _op(host):
    return getattr(host, host.__name__.split(".")[-2])


def _no_kernel(key: tuple) -> str:
    return (_ROUTES if len(key) == len(_DIMS) else Routes(OP, _ROUTES.hosts)).blocker(key)


def _host(key: tuple):
    host = _ROUTES.host(key)
    if host is None:
        raise ValueError(_no_kernel(key))
    return host


def bf16_latent_down_mxfp4_quant_shapes() -> frozenset[tuple[int, int]]:
    """The covered (N, K) of the weight."""
    return _ROUTES.shapes()


def bf16_latent_down_mxfp4_quant(x: torch.Tensor, w: torch.Tensor, out: tuple | None = None, *,
                                 shuffled: bool = False, out_bf16: torch.Tensor | None = None):
    """(codes, scales) = MXFP4 of bf16(x @ W^T) from the routed package (its docstring is the contract)."""
    return _op(_host(tuple(w.shape)))(x, w, out, shuffled=shuffled, out_bf16=out_bf16)


def bf16_latent_down_mxfp4_quant_weight_reason(w: torch.Tensor, bias: torch.Tensor | None = None) -> str | None:
    """None if a layer with this weight (and bias) can run the op, else why not (load time, no activations)."""
    if not routes.available():
        return f"{OP}: needs gfx950 with FlyDSL"
    host = _ROUTES.host(tuple(w.shape))
    if host is None:
        return _no_kernel(tuple(w.shape))
    if not host.supports(w, bias):
        return f"{OP}: operand layout ({host.__package__}.supports)"
    return None


def bf16_latent_down_mxfp4_quant_blocker(x: torch.Tensor, w: torch.Tensor, out: tuple | None = None, *,
                                         shuffled: bool = False, out_bf16: torch.Tensor | None = None) -> str | None:
    """None if ``bf16_latent_down_mxfp4_quant`` serves this call, else why not (host checks only)."""
    reason = bf16_latent_down_mxfp4_quant_weight_reason(w)
    if reason is not None:
        return reason
    n, k = w.shape
    if x.ndim != 2 or x.shape[1] != k or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.device != w.device:
        return f"{OP}: x must be contiguous bf16 [rows, {k}] on the weight's device"
    m = x.shape[0]
    if out is not None:
        codes, scales = out
        if (tuple(codes.shape) != (m, n // 2) or tuple(scales.shape) != (m, n // 32)
                or codes.element_size() != 1 or scales.element_size() != 1
                or not codes.is_contiguous() or not scales.is_contiguous()
                or codes.device != x.device or scales.device != x.device):
            return f"{OP}: out must be contiguous 1-byte (codes [{m}, {n // 2}], scales [{m}, {n // 32}]) on x's device"
    if out_bf16 is not None and (tuple(out_bf16.shape) != (m, n) or out_bf16.dtype != torch.bfloat16
                                 or not out_bf16.is_contiguous() or out_bf16.device != x.device):
        return f"{OP}: out_bf16 must be contiguous bf16 [{m}, {n}] on x's device"
    return None


def latent_down_xcd_local(device_index: int, on: bool, *, w: torch.Tensor) -> None:
    """Set whether the K slices of a tile exchange partials through one L2 on this device (the package probes the
    placement on its first launch; a caller running another kernel beside it turns it off)."""
    _host(tuple(w.shape))._XCD_LOCAL[device_index] = bool(on)


def latent_down_swap_stream_workspaces(device_index: int, stream_a: int, stream_b: int) -> None:
    """Exchange the per-stream workspaces (partial slabs, tile tickets) of two streams on a device, dict keys only:
    see ``latent_moe_common.swap_stream_workspaces``."""
    for host in [sys.modules[n] for n in _ROUTES.hosts.values() if n in sys.modules]:
        work = host._WORK
        moved = {}
        for key in [k for k in work if k[0] == device_index and k[1] in (stream_a, stream_b)]:
            moved[(key[0], stream_b if key[1] == stream_a else stream_a, *key[2:])] = work.pop(key)
        work.update(moved)


bf16_latent_down_mxfp4_quant_unsupported_reason = renamed(bf16_latent_down_mxfp4_quant_blocker,
                                                          "bf16_latent_down_mxfp4_quant_unsupported_reason")
