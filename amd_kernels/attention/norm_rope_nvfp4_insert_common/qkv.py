# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The attention input of an NVFP4 sparse layer (q / k / index norms + RoPE, the NVFP4 K / V and index-K inserts) from
its QKV rows or from the MXFP8 codes of the QKV GEMM's input: routes a call by (head_dim, rotary_dim, page) of its q
norm weight, cos / sin cache and KV cache to the shape package (``_ROUTES``, imported on the first call of its key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes

_NAMES = ("head_dim", "rotary_dim", "page")
_HOSTS = {(128, 64, 128): "amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128"}
_ROUTES = Routes("qkv_norm_rope_nvfp4_insert", _HOSTS, _NAMES)
_GEMM = Routes("mxfp8_gemm_qkv_norm_rope_nvfp4_insert", _HOSTS, _NAMES)
_LAYER = Routes("norm_rope_nvfp4_insert_layer", _HOSTS, _NAMES)
_PAGE_BYTES = 72  # one NVFP4 token row of 128 dims: 64 B of codes + 8 B of scales


def _key(q_norm_weight, cos_sin_cache, kv_cache):
    return (q_norm_weight.numel(), cos_sin_cache.shape[-1], kv_cache.shape[2] if kv_cache.dim() == 4 else 0)


def qkv_norm_rope_nvfp4_insert_shapes() -> frozenset[tuple[int, int, int]]:
    """The (head_dim, rotary_dim, page) with a kernel."""
    return _ROUTES.shapes()


def norm_rope_nvfp4_insert_layer_blocker(num_heads: int, num_kv_heads: int, num_index_heads: int, rotary_dim: int,
                                         *, head_dim: int, index_head_dim: int, block_size: int, neox: bool = True,
                                         index_q_dtype: torch.dtype = torch.float8_e4m3fn,
                                         cos_sin_cache: torch.Tensor | None = None) -> str | None:
    """None when a sparse attention layer (this rank's heads, its RoPE, the index-q dtype) fits the ops of this
    family: a load-time check (the ops cannot see the RoPE style)."""
    key = (head_dim, rotary_dim, block_size)
    reason = _LAYER.blocker(key)
    if reason is not None:
        return reason
    reason = _LAYER.package(key).insert_blocker(
        num_heads, num_kv_heads, num_index_heads, rotary_dim, head_dim=head_dim, index_head_dim=index_head_dim,
        neox=neox, index_q_dtype=index_q_dtype, cos_sin_cache=cos_sin_cache)
    return None if reason is None else f"{_LAYER.op}: {reason}"


def qkv_norm_rope_nvfp4_insert(qkv, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
                               cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache, index_cache,
                               k_scale, v_scale, *, num_heads, num_kv_heads, num_index_heads, eps,
                               index_q_dtype=torch.float8_e4m3fn):
    """(q [n, nq * d] bf16, index-q [n, niq * d]) of the bf16 QKV rows, K / V and index-K into their NVFP4 pages (the
    routed package's ``qkv_norm_rope_nvfp4_insert``)."""
    return _ROUTES.package(_key(q_norm_weight, cos_sin_cache, kv_cache)).qkv_norm_rope_nvfp4_insert(
        qkv, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions,
        slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, num_heads=num_heads,
        num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, eps=eps, index_q_dtype=index_q_dtype)


def mxfp8_gemm_qkv_norm_rope_nvfp4_insert(activation, activation_scale, weight, weight_scale, m, out_dtype,
                                          q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
                                          cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache,
                                          index_cache, k_scale, v_scale, *, num_heads, num_kv_heads, num_index_heads,
                                          eps, index_q_dtype=torch.float8_e4m3fn, one_launch=True):
    """``qkv_norm_rope_nvfp4_insert`` of the QKV GEMM of ``m`` rows from its MXFP8 input (``gemm.mxfp8_common``'s
    pre-quantized codes and scales, preshuffled weight): the routed package's ``mxfp8_gemm_qkv_norm_rope_nvfp4_insert``
    (one launch where it applies and ``one_launch`` allows it)."""
    return _GEMM.package(_key(q_norm_weight, cos_sin_cache, kv_cache)).mxfp8_gemm_qkv_norm_rope_nvfp4_insert(
        activation, activation_scale, weight, weight_scale, m, out_dtype, q_norm_weight, k_norm_weight,
        index_q_norm_weight, index_k_norm_weight, cos_sin_cache, positions, slot_mapping, index_slot_mapping,
        kv_cache, index_cache, k_scale, v_scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        num_index_heads=num_index_heads, eps=eps, index_q_dtype=index_q_dtype, one_launch=one_launch)


def _insert_reason(op, routes_, n, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
                   cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache, index_cache, k_scale, v_scale,
                   num_heads, num_kv_heads, num_index_heads, eps, index_q_dtype):
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _key(q_norm_weight, cos_sin_cache, kv_cache)
    reason = routes_.blocker(key)
    if reason is not None:
        return reason
    d, rot, page = key
    reason = routes_.package(key).insert_blocker(
        num_heads, num_kv_heads, num_index_heads, rot, head_dim=d, index_head_dim=index_q_norm_weight.numel(),
        index_q_dtype=index_q_dtype, cos_sin_cache=cos_sin_cache)
    if reason is not None:
        return f"{op}: {reason}"
    for name, w in (("q", q_norm_weight), ("k", k_norm_weight), ("index q", index_q_norm_weight),
                    ("index k", index_k_norm_weight)):
        if w.dtype != torch.bfloat16 or w.numel() != d or not w.is_contiguous():
            return f"{op}: {name} norm weight {tuple(w.shape)} {w.dtype} (contiguous bf16 [{d}])"
    if not eps >= 2.0**-126:
        return f"{op}: eps {eps} (a normal fp32, >= 2^-126)"
    for name, t in (("positions", positions), ("slot mapping", slot_mapping),
                    ("index slot mapping", index_slot_mapping)):
        if t.dtype != torch.int64 or not t.is_contiguous() or t.shape[0] < n:
            return f"{op}: {name} {tuple(t.shape)} {t.dtype} (contiguous int64 [{n}])"
    c, ic = kv_cache, index_cache
    if (c.dtype != torch.uint8 or tuple(c.shape[1:]) != (2 * num_kv_heads, page, _PAGE_BYTES)
            or c.stride(1) != page * _PAGE_BYTES or c.stride(3) != 1):
        return f"{op}: KV cache {tuple(c.shape)} {c.dtype} (uint8 [pages, {2 * num_kv_heads}, {page}, 72], packed)"
    if (ic.dtype != torch.uint8 or ic.dim() != 3 or tuple(ic.shape[1:]) != (page, _PAGE_BYTES)
            or ic.stride(1) != _PAGE_BYTES or ic.stride(2) != 1):
        return f"{op}: index cache {tuple(ic.shape)} {ic.dtype} (uint8 [pages, {page}, 72], packed)"
    if not (k_scale is not None and v_scale is not None and k_scale.dtype == v_scale.dtype == torch.float32):
        return f"{op}: fp32 k / v scales"
    return None


def _width(q_norm_weight, num_heads, num_kv_heads, num_index_heads):
    return (num_heads + 2 * num_kv_heads + num_index_heads + 1) * q_norm_weight.numel()


def qkv_norm_rope_nvfp4_insert_blocker(qkv, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
                                       cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache,
                                       index_cache, k_scale, v_scale, *, num_heads, num_kv_heads, num_index_heads,
                                       eps, index_q_dtype=torch.float8_e4m3fn) -> str | None:
    """None when ``qkv_norm_rope_nvfp4_insert`` serves this call, else why not (host checks only)."""
    op = _ROUTES.op
    n = qkv.shape[0] if qkv.dim() == 2 else -1
    reason = _insert_reason(op, _ROUTES, n, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
                            cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache, index_cache,
                            k_scale, v_scale, num_heads, num_kv_heads, num_index_heads, eps, index_q_dtype)
    if reason is not None:
        return reason
    w = _width(q_norm_weight, num_heads, num_kv_heads, num_index_heads)
    if qkv.dim() != 2 or qkv.dtype != torch.bfloat16 or qkv.shape[1] != w or not qkv.is_contiguous():
        return f"{op}: qkv {tuple(qkv.shape)} {qkv.dtype} (contiguous bf16 [n, {w}])"
    return None


def mxfp8_gemm_qkv_norm_rope_nvfp4_insert_blocker(activation, activation_scale, weight, weight_scale, m, out_dtype,
                                                  q_norm_weight, k_norm_weight, index_q_norm_weight,
                                                  index_k_norm_weight, cos_sin_cache, positions, slot_mapping,
                                                  index_slot_mapping, kv_cache, index_cache, k_scale, v_scale, *,
                                                  num_heads, num_kv_heads, num_index_heads, eps,
                                                  index_q_dtype=torch.float8_e4m3fn, one_launch=True) -> str | None:
    """None when ``mxfp8_gemm_qkv_norm_rope_nvfp4_insert`` serves this call, else why not (host checks only; the
    MXFP8 operands in ``gemm.mxfp8_common``'s layouts are the caller's contract)."""
    op = _GEMM.op
    reason = _insert_reason(op, _GEMM, m, q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight,
                            cos_sin_cache, positions, slot_mapping, index_slot_mapping, kv_cache, index_cache,
                            k_scale, v_scale, num_heads, num_kv_heads, num_index_heads, eps, index_q_dtype)
    if reason is not None:
        return reason
    w = _width(q_norm_weight, num_heads, num_kv_heads, num_index_heads)
    if weight.dim() != 2 or weight.shape[0] != w or weight.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return f"{op}: weight {tuple(weight.shape)} {weight.dtype} (MXFP8 [{w}, K])"
    if out_dtype != torch.bfloat16:
        return f"{op}: out dtype {out_dtype} (bf16)"
    if m < 0 or activation.dim() != 2 or activation.shape[1] != weight.shape[1]:
        return f"{op}: activation {tuple(activation.shape)} for {m} rows (codes [>= m, {weight.shape[1]}])"
    return None
