# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DSA prefill indexer: MXFP4 logits of ragged rows (routed by heads, head dim) and the exact per-row top-k over
each row's key window (routed by top-k)."""

import functools
import importlib

import torch

from .device import available

_SCORE_HOSTS = {(32, 128): "amd_kernels.indexer.mxfp4_prefill_score_d128_h32"}
_TOPK_HOSTS = {(2048,): "amd_kernels.indexer.mxfp4_prefill_topk_d128_h32_top2048"}


@functools.cache
def _score_host(key):
    name = _SCORE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _topk_host(key):
    name = _TOPK_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


@functools.cache
def _score_max_out_bytes(key) -> int:
    return importlib.import_module(_SCORE_HOSTS[key] + ".kernel").MAX_OUT_BYTES


def _no_kernel(op, dims, hosts) -> str:
    return f"{op}: no kernel for {dims} (covered: {', '.join(map(str, sorted(hosts)))})"


def _score_key(q_payload):
    return (q_payload.shape[1], 2 * q_payload.shape[2]) if q_payload.ndim == 3 else None


def _score_route(q_payload):
    key = _score_key(q_payload)
    host = _score_host(key)
    if host is None:
        raise ValueError(_no_kernel("mxfp4_prefill_score", _score_dims(key), _SCORE_HOSTS))
    return host


def _score_dims(key) -> str:
    return "this q_payload rank" if key is None else f"heads={key[0]}, d={key[1]}"


def mxfp4_prefill_score(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out, weight_scale=1.0):
    """The logits of every row's key window ``[ks, ke)`` into ``out`` (the routed package's ``score_rows``)."""
    return _score_route(q_payload).score_rows(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out,
                                              weight_scale)


def mxfp4_prefill_score_shapes() -> frozenset:
    """The covered (heads, d)."""
    return frozenset(_SCORE_HOSTS)


def mxfp4_prefill_score_unsupported_reason(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out,
                                           weight_scale=1.0) -> str | None:
    """None when ``mxfp4_prefill_score`` serves this call, else why not (host checks only)."""
    op = "mxfp4_prefill_score"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _score_key(q_payload)
    if _score_host(key) is None:
        return _no_kernel(op, _score_dims(key), _SCORE_HOSTS)
    heads, d = key
    rows = q_payload.shape[0]
    if (q_payload.dtype != torch.uint8 or not q_payload.is_contiguous() or q_scale.dtype != torch.uint8
            or not q_scale.is_contiguous() or q_scale.shape[0] != rows or q_scale.numel() != rows * heads * d // 16):
        return f"{op}: q_payload and q_scale must be contiguous uint8, q_scale {heads * d // 16} bytes a row"
    if weights.dtype != torch.bfloat16 or weights.shape != (rows, heads) or not weights.is_contiguous():
        return f"{op}: weights must be contiguous bf16 [rows, {heads}]"
    if (key_payload.dtype != torch.uint8 or key_payload.shape[1:] != (1, 4, 64, d // 8)
            or not key_payload.is_contiguous() or key_scale.dtype != torch.uint8 or not key_scale.is_contiguous()
            or key_scale.shape != key_payload.shape[:4]):
        return (f"{op}: key_payload must be contiguous uint8 [pages, 1, 4, 64, {d // 8}], key_scale contiguous uint8 "
                "[pages, 1, 4, 64]")
    if (ks.dtype != torch.int32 or ke.dtype != torch.int32 or ks.shape != (rows,) or ke.shape != (rows,)
            or not ks.is_contiguous() or not ke.is_contiguous()):
        return f"{op}: ks and ke must be contiguous int32 [rows]"
    if out.dtype != torch.float32 or out.ndim != 2 or out.stride(1) != 1:
        return f"{op}: out must be fp32 [rows, width] with unit column stride"
    if rows and (out.shape[0] != rows or out.shape[1] < key_payload.shape[0] * 64):
        return f"{op}: out must be [rows, >= pages * 64]"
    if out.numel() * 4 > _score_max_out_bytes(key):
        return f"{op}: logits of {out.numel() * 4} bytes (the kernel takes <= {_score_max_out_bytes(key)})"
    operands = (q_payload, q_scale, weights, key_payload, key_scale, ks, ke)
    device = out.device
    if device.type == "cpu" or any(t.device != device for t in operands):
        return f"{op}: operands must be on one GPU"
    return None


def _topk_key(out):
    return (out.shape[1],) if out.ndim == 2 else None


def _topk_dims(key) -> str:
    return "this out rank" if key is None else f"topk={key[0]}"


def _topk_route(out):
    key = _topk_key(out)
    host = _topk_host(key)
    if host is None:
        raise ValueError(_no_kernel("mxfp4_prefill_topk", _topk_dims(key), _TOPK_HOSTS))
    return host


def mxfp4_prefill_topk(logits, ks, ke, out, order=None):
    """The window-relative positions of every row's top-k logits of ``[ks, ke)`` into ``out`` (the routed package's
    ``topk_rows``)."""
    # the package's warm-up inputs take no ``order``: passed only when given
    return _topk_route(out).topk_rows(logits, ks, ke, out, **({"order": order} if order is not None else {}))


def mxfp4_prefill_topk_shapes() -> frozenset:
    """The covered (top-k,)."""
    return frozenset(_TOPK_HOSTS)


def mxfp4_prefill_topk_unsupported_reason(logits, ks, ke, out, order=None) -> str | None:
    """None when ``mxfp4_prefill_topk`` serves this call, else why not (host checks only)."""
    op = "mxfp4_prefill_topk"
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _topk_key(out)
    if _topk_host(key) is None:
        return _no_kernel(op, _topk_dims(key), _TOPK_HOSTS)
    if logits.ndim != 2 or logits.dtype != torch.float32 or logits.stride(1) != 1 or logits.stride(0) % 4:
        return f"{op}: logits must be fp32 [rows, width], unit column stride, row stride a multiple of 4"
    if logits.data_ptr() % 16:
        return f"{op}: logits must be 16-byte aligned"
    rows = logits.shape[0]
    if (ks.dtype != torch.int32 or ke.dtype != torch.int32 or ks.shape != (rows,) or ke.shape != (rows,)
            or not ks.is_contiguous() or not ke.is_contiguous()):
        return f"{op}: ks and ke must be contiguous int32 [rows]"
    if out.dtype != torch.int32 or out.shape[0] != rows or out.stride(1) != 1:
        return f"{op}: out must be int32 [rows, {key[0]}] with unit column stride"
    if order is not None and (order.dtype != torch.int32 or order.shape != (rows,) or not order.is_contiguous()):
        return f"{op}: order must be contiguous int32 [rows]"
    operands = (logits, ks, ke) if order is None else (logits, ks, ke, order)
    device = out.device
    if device.type == "cpu" or any(t.device != device for t in operands):
        return f"{op}: operands must be on one GPU"
    return None


def prefill_topk_workspace_bytes(rows: int, width: int, *, topk: int) -> int:
    """Scratch bytes of one ``mxfp4_prefill_topk`` call over ``rows`` rows of ``width`` logits."""
    host = _topk_host((topk,))
    if host is None:
        raise ValueError(_no_kernel("mxfp4_prefill_topk", f"topk={topk}", _TOPK_HOSTS))
    return host.workspace_bytes(rows, width)
