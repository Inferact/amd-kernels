# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The DSA decode indexer's MXFP4 paged scorer + per-row top-k, routed by (heads, head dim, top-k, page rows).

The scratch helpers (``paged_score_topk_prefer`` / ``_preallocate`` / ``_reset_counts``) run before the query rows
exist: they route by (top-k, page rows) of the output and the index cache.
"""

import functools
import importlib

import torch

from .device import available

_OP = "mxfp4_paged_score_topk"
_DIMS = ("heads", "d", "topk", "page_rows")
_SHAPE_HOSTS = {(32, 128, 2048, 64): "amd_kernels.indexer.mxfp4_paged_score_topk_d128_h32_top2048"}


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


def _key(q_payload, out_indices, cache):
    if q_payload.ndim != 3 or out_indices.ndim != 2 or cache.ndim != 3:
        return None
    return (q_payload.shape[1], 2 * q_payload.shape[2], out_indices.shape[1], cache.shape[1])


def _no_kernel(key) -> str:
    dims = "these operand ranks" if key is None else ", ".join(f"{n}={v}" for n, v in zip(_DIMS, key, strict=True))
    return f"{_OP}: no kernel for {dims} (covered: {', '.join(map(str, sorted(_SHAPE_HOSTS)))})"


def _route(q_payload, out_indices, cache):
    key = _key(q_payload, out_indices, cache)
    host = _shape_host(key)
    if host is None:
        raise ValueError(_no_kernel(key))
    return host


@functools.cache
def _tail_host(tail):
    keys = [k for k in _SHAPE_HOSTS if k[2:] == tail]
    return _shape_host(keys[0]) if len(keys) == 1 else None


def _scratch_host(cache, out_indices, need: bool):
    host = _tail_host((out_indices.shape[-1], cache.shape[1])) if cache.ndim == 3 else None
    if host is None and need:
        raise ValueError(f"{_OP}: no kernel for topk={out_indices.shape[-1]}, cache {tuple(cache.shape)}")
    return host


def mxfp4_paged_score_topk(q_payload, q_scales, weights, cache, block_table, query_start_loc, row_ends, max_query_len,
                           out_indices, out_scores, uniform=False) -> None:
    """Scores and per-row top-k of a decode batch in one launch (the routed package's ``mxfp4_paged_score_topk``)."""
    # the package's warm-up inputs take no ``uniform``: passed only when set
    _route(q_payload, out_indices, cache).mxfp4_paged_score_topk(
        q_payload, q_scales, weights, cache, block_table, query_start_loc, row_ends, max_query_len, out_indices,
        out_scores, **({"uniform": True} if uniform else {}))


def mxfp4_paged_score_topk_kernel(q_payload, q_scales, weights, cache, block_table, query_start_loc, row_ends,
                                  max_query_len, out_indices, out_scores, uniform=False):
    """The routed ``amd_kernels.warm_up`` entry (``.entry``), for a caller that compiles without launching."""
    return _route(q_payload, out_indices, cache).mxfp4_paged_score_topk


def mxfp4_paged_score_topk_shapes() -> frozenset:
    """The covered (heads, d, top-k, page rows)."""
    return frozenset(_SHAPE_HOSTS)


def mxfp4_paged_score_topk_unsupported_reason(q_payload, q_scales, weights, cache, block_table, query_start_loc,
                                              row_ends, max_query_len, out_indices, out_scores,
                                              uniform=False) -> str | None:
    """None when ``mxfp4_paged_score_topk`` serves this call, else why not (host checks only)."""
    if not available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(q_payload, out_indices, cache)
    pkg = _shape_host(key)
    if pkg is None:
        return _no_kernel(key)
    heads, d, topk, page = key
    rows, batch = q_payload.shape[0], query_start_loc.numel() - 1
    geo, host = pkg.geometry, pkg.host
    if not 1 <= rows <= geo.MAX_ROWS:
        return f"{_OP}: {rows} rows (the kernel takes 1..{geo.MAX_ROWS})"
    if not 1 <= max_query_len <= geo.QMAX:
        return f"{_OP}: max_query_len {max_query_len} (the kernel takes 1..{geo.QMAX})"
    if (batch < 1 or query_start_loc.dtype != torch.int32 or query_start_loc.ndim != 1
            or not query_start_loc.is_contiguous()):
        return f"{_OP}: query_start_loc must be contiguous int32 [sequences + 1], sequences >= 1"
    if (block_table.ndim != 2 or block_table.dtype != torch.int32 or block_table.stride(1) != 1
            or block_table.shape[0] < batch):
        return f"{_OP}: block_table must be int32 [>= sequences, blocks] with unit column stride"
    width = block_table.shape[1] * page
    if width > host.TIER_WIDTHS[-1]:
        return f"{_OP}: block table of {width} keys (the kernel takes <= {host.TIER_WIDTHS[-1]})"
    tile = host.plan(max_query_len, batch)[0]
    if batch * -(-max_query_len // tile) > geo.COUNTER_GROUPS:
        return f"{_OP}: {batch} sequences (the kernel takes <= {geo.COUNTER_GROUPS} row tiles)"
    row_bytes = d // 2 + d // 32
    if (cache.dtype != torch.uint8 or cache.shape[2] != row_bytes or cache.stride(2) != 1
            or cache.stride(1) != row_bytes or cache.stride(0) < page * row_bytes):
        return f"{_OP}: index cache must be uint8 [pages, {page}, {row_bytes}], page stride >= {page * row_bytes}"
    if (q_payload.dtype != torch.uint8 or not q_payload.is_contiguous() or q_scales.dtype != torch.uint8
            or not q_scales.is_contiguous() or q_scales.shape[0] != rows or q_scales.numel() != rows * heads * d // 16):
        return f"{_OP}: q_payload and q_scales must be contiguous uint8, q_scales {heads * d // 16} bytes a row"
    if weights.dtype != torch.float32 or weights.shape != (rows, heads) or not weights.is_contiguous():
        return f"{_OP}: weights must be contiguous fp32 [rows, {heads}]"
    if row_ends.dtype != torch.int32 or row_ends.shape != (rows,) or not row_ends.is_contiguous():
        return f"{_OP}: row_ends must be contiguous int32 [rows]"
    if (out_indices.dtype != torch.int32 or out_indices.shape[0] != rows or out_indices.stride(1) != 1
            or out_scores.dtype != torch.float32 or out_scores.shape != (rows, topk) or out_scores.stride(1) != 1):
        return f"{_OP}: out_indices int32 / out_scores fp32 must be [rows, {topk}] with unit column stride"
    operands = (q_payload, q_scales, weights, block_table, query_start_loc, row_ends, out_indices, out_scores)
    device = cache.device
    if device.type == "cpu" or any(t.device != device for t in operands):
        return f"{_OP}: operands must be on the index cache's GPU"
    return None


def paged_score_topk_prefer(batch: int, width: int, *, cache, out_indices) -> bool:
    """Whether the fused call beats the unfused scorer + top-k for ``batch`` sequences of ``width`` keys (the routed
    package's ``prefer_fused``; False off the table)."""
    host = _scratch_host(cache, out_indices, need=False)
    return host is not None and host.prefer_fused(batch, width)


def paged_score_topk_preallocate(device, streams, *, cache, out_indices) -> None:
    """Create the scratch of each of ``streams`` (one set a device and stream) before graph capture."""
    host = _scratch_host(cache, out_indices, need=True).host
    for stream in streams:
        host._scratch_set(torch.device(device), stream.cuda_stream)


def paged_score_topk_reset_counts(query_start_loc, max_query_len, *, cache, out_indices) -> None:
    """Zero the ready counts of the (sequence, row tile) groups a call over this batch uses, in the current stream's
    scratch: a group without rows (a padding sequence, a tile past a short sequence) has no merge workgroup to reset
    its count, and a later call in which it has rows would pass its wait early. Graph-safe after the preallocate."""
    pkg = _scratch_host(cache, out_indices, need=True)
    batch = query_start_loc.numel() - 1
    tile = pkg.host.plan(max_query_len, batch)[0]
    groups = batch * -(-max_query_len // tile)
    pkg.host._scratch(cache.device)[4][: groups * pkg.geometry.COUNTER_BYTES // 4].zero_()
