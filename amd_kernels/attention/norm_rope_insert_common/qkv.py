# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One attention layer's q / k RMS norm + RoPE + K / V insert: routes a call by (head_dim, rotary_dim) of its norm
weight and cos / sin cache to the shape package (``_ROUTES``, imported on the first call of its key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

_NAMES = ("head_dim", "rotary_dim")
_ROUTES = Routes("qkv_norm_rope_insert", {(64, 64): "amd_kernels.attention.norm_rope_insert_d64_rot64"}, _NAMES)
_ROPE = Routes("rope", _ROUTES.hosts, _NAMES)
_OP = _ROUTES.op


def _op_module(key):
    return _ROUTES.module(key, "qkv_norm_rope_insert")


def _key(norm_weight, cos_sin_cache):
    return (norm_weight.numel(), cos_sin_cache.shape[-1])


def qkv_norm_rope_insert_shapes() -> frozenset[tuple[int, int]]:
    """The (head_dim, rotary_dim) with a kernel."""
    return _ROUTES.shapes()


def norm_rope_insert_rope_blocker(head_size: int, rotary_dim: int, is_neox: bool,
                                  cos_sin_cache: torch.Tensor) -> str | None:
    """None when a layer's RoPE (head size, rotary dims, NeoX or not, its cos / sin cache) fits a kernel of this
    family, else why not: a load-time check (the ops cannot see the RoPE style)."""
    key = (head_size, rotary_dim)
    reason = _ROPE.blocker(key)
    if reason is not None:
        return reason
    reason = _ROPE.package(key).rope_blocker(head_size, rotary_dim, is_neox, cos_sin_cache)
    return None if reason is None else f"rope: {reason}"


def qkv_norm_rope_insert_kernel(qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads,
                                num_kv_heads, q_out, kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None):
    """The routed ``amd_kernels.warm_up`` entry (to compile without launching)."""
    return _op_module(_key(q_norm_weight, cos_sin_cache)).qkv_norm_rope_insert


def qkv_norm_rope_insert(qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads,
                         q_out, kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None):
    """q_out = RoPE(norm(q)) of one layer's fused qkv rows; with ``kv_cache``, RoPE(norm(k)) and v of the tokens with
    slots go into the cache (the routed package's ``qkv_norm_rope_insert``)."""
    _op_module(_key(q_norm_weight, cos_sin_cache)).qkv_norm_rope_insert(qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads,
                                         num_kv_heads, q_out, kv_cache, slot_mapping, k_scale, v_scale)


def qkv_norm_rope_insert_blocker(qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads,
                                 num_kv_heads, q_out, kv_cache=None, slot_mapping=None, k_scale=None,
                                 v_scale=None) -> str | None:
    """None when ``qkv_norm_rope_insert`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(q_norm_weight, cos_sin_cache)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    d = key[0]
    nq, nkv = num_heads, num_kv_heads
    n = qkv.shape[0] if qkv.dim() == 2 else -1
    if (qkv.dim() != 2 or qkv.dtype != torch.bfloat16 or qkv.shape[1] != (nq + 2 * nkv) * d or qkv.stride(1) != 1
            or qkv.stride(0) % 8):
        return (f"{_OP}: qkv {tuple(qkv.shape)} {qkv.dtype} stride {qkv.stride()} (bf16 [n, {(nq + 2 * nkv) * d}], "
                "unit column stride, row stride a multiple of 8)")
    if positions.dtype != torch.int64 or not positions.is_contiguous() or positions.shape[0] < n:
        return f"{_OP}: positions {tuple(positions.shape)} {positions.dtype} (contiguous int64 [{n}])"
    for name, w in (("q", q_norm_weight), ("k", k_norm_weight)):
        if w.dtype != torch.bfloat16 or w.numel() != d or not w.is_contiguous():
            return f"{_OP}: {name} norm weight {tuple(w.shape)} {w.dtype} (contiguous bf16 [{d}])"
    cs = cos_sin_cache
    if cs.dtype != torch.bfloat16 or cs.dim() != 2 or not cs.is_contiguous():
        return f"{_OP}: cos / sin cache {tuple(cs.shape)} {cs.dtype} (contiguous bf16 [rows, {key[1]}])"
    if q_out.dtype != torch.bfloat16 or not q_out.is_contiguous() or tuple(q_out.shape) != (n, nq * d):
        return f"{_OP}: q_out {tuple(q_out.shape)} {q_out.dtype} (contiguous bf16 [{n}, {nq * d}])"
    if kv_cache is None or slot_mapping is None:
        return None
    kind = _op_module(key).cache_kind(kv_cache)
    if kind is None or kv_cache.shape[1] != nkv:
        return (f"{_OP}: KV cache {tuple(kv_cache.shape)} {kv_cache.dtype} stride {kv_cache.stride()} (bf16 or e4m3 "
                f"[pages, {nkv}, page, {2 * d}], unit last stride)")
    if slot_mapping.dtype != torch.int64 or not slot_mapping.is_contiguous():
        return f"{_OP}: slot mapping {slot_mapping.dtype} (contiguous int64)"
    if kind == "fp8" and not (k_scale is not None and v_scale is not None
                              and k_scale.dtype == v_scale.dtype == torch.float32):
        return f"{_OP}: an e4m3 cache needs fp32 k / v scales"
    return None


qkv_norm_rope_insert_unsupported_reason = renamed(qkv_norm_rope_insert_blocker,
                                                  "qkv_norm_rope_insert_unsupported_reason")
rope_unsupported_reason = renamed(norm_rope_insert_rope_blocker, "rope_unsupported_reason")
