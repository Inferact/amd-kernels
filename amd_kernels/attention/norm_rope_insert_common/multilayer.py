# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The context rows' K norm + RoPE + K / V inserts into every layer's cache: routes a call by (head_dim, rotary_dim)
of its K norm weights and cos / sin cache to the shape package (``_ROUTES``, imported on the first call of its
key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

_ROUTES = Routes("kv_norm_rope_insert_multilayer", {(64, 64): "amd_kernels.attention.norm_rope_insert_d64_rot64"},
                 ("head_dim", "rotary_dim"))
_OP = _ROUTES.op


def _op_module(key):
    return _ROUTES.module(key, "kv_norm_rope_insert_multilayer")


def _key(k_norm_weight, cos_sin_cache):
    return (k_norm_weight.shape[-1], cos_sin_cache.shape[-1])


def kv_norm_rope_insert_multilayer_shapes() -> frozenset[tuple[int, int]]:
    """The (head_dim, rotary_dim) with a kernel."""
    return _ROUTES.shapes()


def kv_norm_rope_insert_multilayer_kernel(kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings,
                                          k_scales, v_scales, eps, n_insert=None):
    """The routed ``amd_kernels.warm_up`` entry (to compile without launching)."""
    return _op_module(_key(k_norm_weight, cos_sin_cache)).kv_norm_rope_insert_multilayer


def kv_norm_rope_insert_multilayer(kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales,
                                   v_scales, eps, n_insert=None):
    """K norm + RoPE and the K / V inserts of the context rows' fused K / V row into every layer's cache (the routed
    package's ``kv_norm_rope_insert_multilayer``)."""
    _op_module(_key(k_norm_weight, cos_sin_cache)).kv_norm_rope_insert_multilayer(
        kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales, v_scales, eps,
        n_insert=n_insert)


def kv_norm_rope_insert_multilayer_blocker(kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings,
                                           k_scales, v_scales, eps, n_insert=None) -> str | None:
    """None when ``kv_norm_rope_insert_multilayer`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(k_norm_weight, cos_sin_cache)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    d = key[0]
    if not _op_module(key).supports_multilayer(kv_caches, k_scales, v_scales):
        c = kv_caches[0] if kv_caches else None
        got = "none" if c is None else f"{tuple(c.shape)} {c.dtype} stride {c.stride()}"
        return (f"{_OP}: KV caches {got} (bf16, or e4m3 with fp32 scales, [pages, kv_heads, page, {2 * d}], unit last "
                "stride, every layer alike)")
    nl, nkv = len(kv_caches), kv_caches[0].shape[1]
    n = kv.shape[0] if kv.dim() == 2 else -1
    width = nl * 2 * nkv * d
    if kv.dim() != 2 or kv.dtype != torch.bfloat16 or kv.shape[1] != width or kv.stride(1) != 1 or kv.stride(0) % 8:
        return (f"{_OP}: kv {tuple(kv.shape)} {kv.dtype} stride {kv.stride()} (bf16 [n, {width}], unit column "
                "stride, row stride a multiple of 8)")
    if positions.dtype != torch.int64 or not positions.is_contiguous() or positions.shape[0] < n:
        return f"{_OP}: positions {tuple(positions.shape)} {positions.dtype} (contiguous int64 [{n}])"
    if (k_norm_weight.dtype != torch.bfloat16 or tuple(k_norm_weight.shape) != (nl, d)
            or not k_norm_weight.is_contiguous()):
        return f"{_OP}: K norm weights {tuple(k_norm_weight.shape)} {k_norm_weight.dtype} (contiguous bf16 [{nl}, {d}])"
    cs = cos_sin_cache
    if cs.dtype != torch.bfloat16 or cs.dim() != 2 or not cs.is_contiguous():
        return f"{_OP}: cos / sin cache {tuple(cs.shape)} {cs.dtype} (contiguous bf16 [rows, {key[1]}])"
    m = n if n_insert is None else min(n_insert, n)
    if len(slot_mappings) != nl or any(
        s is None or s.dtype != torch.int64 or s.dim() != 1 or s.stride(0) != 1 for s in slot_mappings
    ):
        return f"{_OP}: slot mappings (one contiguous int64 [rows] a layer, {nl} layers)"
    if len({min(m, s.shape[0]) for s in slot_mappings}) > 1:
        return f"{_OP}: slot mappings of {sorted({s.shape[0] for s in slot_mappings})} rows (every layer the same)"
    blocker = _op_module(key).capture_blocker(kv, kv_caches, k_scales, v_scales)
    return None if blocker is None else f"{_OP}: {blocker}"


kv_norm_rope_insert_multilayer_unsupported_reason = renamed(kv_norm_rope_insert_multilayer_blocker,
                                                            "kv_norm_rope_insert_multilayer_unsupported_reason")
