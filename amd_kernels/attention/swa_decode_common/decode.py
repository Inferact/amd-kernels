# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sliding-window decode attention over e4m3 K / V pages as an o_proj input: routes a call by (head_dim, query heads
per KV head), read from the cache [pages, KV heads, page, 2 * head_dim] and q, to the shape package (``_ROUTES``,
imported on the first call of its key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes

_NAMES = ("head_dim", "heads_per_kv")
_ROUTES = Routes("swa_decode_oproj_input", {(128, 16): "amd_kernels.attention.swa_decode_fp8kv_gqa16_d128"}, _NAMES)
_LAYER = Routes("swa_decode_layer", _ROUTES.hosts, _NAMES)
_OP = _ROUTES.op
_FP8 = (torch.uint8, torch.float8_e4m3fn)
_I32 = torch.int32


def _key(q, kv_cache):
    if kv_cache.dim() != 4 or q.dim() not in (2, 3):
        return None
    d = kv_cache.shape[3] // 2
    heads = q.shape[1] if q.dim() == 3 else q.shape[1] // max(d, 1)
    return (d, heads // max(kv_cache.shape[1], 1))


def _layer_key(head_dim, num_heads, num_kv_heads):
    return (head_dim, num_heads // num_kv_heads if num_kv_heads else 0)


def swa_decode_oproj_input_shapes() -> frozenset[tuple[int, int]]:
    """The (head_dim, query heads per KV head) with a kernel."""
    return _ROUTES.shapes()


def swa_decode_layer_blocker(head_dim: int, num_heads: int, num_kv_heads: int, window: int | None) -> str | None:
    """None when a sliding-window layer (this rank's heads, its window) fits a kernel of this family at up to
    ``swa_decode_max_query_len`` rows per request: a load-time check."""
    key = _layer_key(head_dim, num_heads, num_kv_heads)
    reason = _LAYER.blocker(key)
    if reason is not None:
        return reason
    reason = _LAYER.package(key).layer_blocker(head_dim, num_heads, num_kv_heads, window)
    return None if reason is None else f"{_LAYER.op}: {reason}"


def swa_decode_max_query_len(head_dim: int, num_heads: int, num_kv_heads: int) -> int:
    """The most query rows per request the layer's kernel takes (more: the caller's own attention)."""
    return _LAYER.package(_layer_key(head_dim, num_heads, num_kv_heads)).MAX_QL


def swa_decode_oproj_input(q, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window,
                           causal, k_scale, v_scale, *, oproj_n=None, output=None):
    """The attention of ``q`` ([rows, H, head_dim] or [rows, H * head_dim] bf16) over the window as its o_proj's
    input: (codes, scales, tile) of the MXFP8 epilogue with ``oproj_n`` where the package writes it, else the bf16
    output shaped like ``q`` (the routed package's ``swa_decode_fp8kv_oproj_input``)."""
    return _ROUTES.module(_key(q, kv_cache), "oproj_input").swa_decode_fp8kv_oproj_input(
        q, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale, window, causal, k_scale,
        v_scale, oproj_n=oproj_n, output=output)


def swa_decode_oproj_input_blocker(q, kv_cache, block_table, seq_lens, query_start_loc, max_query_len, sm_scale,
                                   window, causal, k_scale, v_scale, *, oproj_n=None, output=None) -> str | None:
    """None when ``swa_decode_oproj_input`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(q, kv_cache)
    if key is None:
        return f"{_OP}: q {tuple(q.shape)} ([rows, H, d] or [rows, H * d]), KV cache {tuple(kv_cache.shape)} (4-d)"
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    pkg = _ROUTES.package(key)
    d = key[0]
    if q.dtype != torch.bfloat16 or not q.is_contiguous() or q.shape[-1] % d:
        return f"{_OP}: q {tuple(q.shape)} {q.dtype} (contiguous bf16)"
    c = kv_cache
    if c.dtype not in _FP8 or c.stride(3) != 1 or c.stride(2) != 2 * d or c.shape[2] % pkg.CH:
        return (f"{_OP}: KV cache {tuple(c.shape)} {c.dtype} stride {c.stride()} (e4m3 [pages, KV heads, page, "
                f"{2 * d}], packed tokens, page a multiple of {pkg.CH})")
    if not isinstance(causal, bool) or not 1 <= max_query_len <= pkg.MAX_QL:
        return f"{_OP}: max_query_len {max_query_len}, causal {causal!r} (1..{pkg.MAX_QL}, a bool)"
    if not 0 < window <= pkg.max_window(max_query_len):
        return f"{_OP}: window {window} (1..{pkg.max_window(max_query_len)} at {max_query_len} rows)"
    for name, t in (("block table", block_table), ("seq lens", seq_lens), ("query start loc", query_start_loc)):
        if t.dtype != _I32 or t.stride(-1) != 1:
            return f"{_OP}: {name} {t.dtype} (int32, unit last stride)"
    if not (k_scale is not None and v_scale is not None and k_scale.dtype == v_scale.dtype == torch.float32):
        return f"{_OP}: the e4m3 cache needs fp32 k / v scales"
    if output is not None and (output.dtype != torch.bfloat16 or not output.is_contiguous()
                               or output.shape != q.shape):
        return f"{_OP}: output {tuple(output.shape)} {output.dtype} (contiguous bf16 like q)"
    return None
