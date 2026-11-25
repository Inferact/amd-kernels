# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DSA prefill indexer: MXFP4 logits of ragged rows (routed by heads, head dim) and the exact per-row top-k over
each row's key window (routed by top-k)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

_SCORE_ROUTES = Routes("mxfp4_prefill_score", {(32, 128): "amd_kernels.indexer.mxfp4_prefill_score_d128_h32"},
                       ("heads", "d"))
_TOPK_ROUTES = Routes("mxfp4_prefill_topk", {(2048,): "amd_kernels.indexer.mxfp4_prefill_topk_d128_h32_top2048"},
                      ("topk",))


def _score_max_out_bytes(key) -> int:
    return _SCORE_ROUTES.module(key, "kernel").MAX_OUT_BYTES


def _no_kernel(table, key, what) -> str:
    if key is None:
        return f"{table.op}: no kernel for {what} (covered: {sorted(table.hosts)})"
    return table.blocker(key)


def _score_key(q_payload):
    return (q_payload.shape[1], 2 * q_payload.shape[2]) if q_payload.ndim == 3 else None


def _score_route(q_payload):
    key = _score_key(q_payload)
    host = _SCORE_ROUTES.host(key)
    if host is None:
        raise ValueError(_no_kernel(_SCORE_ROUTES, key, "this q_payload rank"))
    return host


def mxfp4_prefill_score(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out, weight_scale=1.0):
    """The logits of every row's key window ``[ks, ke)`` into ``out`` (the routed package's ``score_rows``)."""
    return _score_route(q_payload).score_rows(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out,
                                              weight_scale)


def mxfp4_prefill_score_shapes() -> frozenset:
    """The covered (heads, d)."""
    return _SCORE_ROUTES.shapes()


def mxfp4_prefill_score_blocker(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, out,
                                weight_scale=1.0) -> str | None:
    """None when ``mxfp4_prefill_score`` serves this call, else why not (host checks only)."""
    op = _SCORE_ROUTES.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _score_key(q_payload)
    if key not in _SCORE_ROUTES.hosts:
        return _no_kernel(_SCORE_ROUTES, key, "this q_payload rank")
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


def _topk_route(out):
    key = _topk_key(out)
    host = _TOPK_ROUTES.host(key)
    if host is None:
        raise ValueError(_no_kernel(_TOPK_ROUTES, key, "this out rank"))
    return host


def mxfp4_prefill_topk(logits, ks, ke, out, order=None):
    """The window-relative positions of every row's top-k logits of ``[ks, ke)`` into ``out`` (the routed package's
    ``topk_rows``)."""
    # the package's warm-up inputs take no ``order``: passed only when given
    return _topk_route(out).topk_rows(logits, ks, ke, out, **({"order": order} if order is not None else {}))


def mxfp4_prefill_topk_shapes() -> frozenset:
    """The covered (top-k,)."""
    return _TOPK_ROUTES.shapes()


def mxfp4_prefill_topk_blocker(logits, ks, ke, out, order=None) -> str | None:
    """None when ``mxfp4_prefill_topk`` serves this call, else why not (host checks only)."""
    op = _TOPK_ROUTES.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _topk_key(out)
    if key not in _TOPK_ROUTES.hosts:
        return _no_kernel(_TOPK_ROUTES, key, "this out rank")
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
    return _TOPK_ROUTES.package((topk,)).workspace_bytes(rows, width)


mxfp4_prefill_score_unsupported_reason = renamed(mxfp4_prefill_score_blocker, "mxfp4_prefill_score_unsupported_reason")
mxfp4_prefill_topk_unsupported_reason = renamed(mxfp4_prefill_topk_blocker, "mxfp4_prefill_topk_unsupported_reason")
