# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunked prefill: routes a call by its recurrent state cache's (heads, head_dim) to the shape package
(``_SHAPE_HOSTS``, imported on the first call of its key)."""

import functools
import importlib

import torch

from . import device as _dev

_SHAPE_HOSTS = {(12, 128): "amd_kernels.attention.kda_prefill_h12_d128"}
_OP = "kda_prefill"


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


def _key(state_cache):
    return (state_cache.shape[1], state_cache.shape[3]) if state_cache.dim() == 4 else None


def _no_kernel(op, state_cache):
    if state_cache.dim() != 4:
        return f"{op}: state cache {tuple(state_cache.shape)} (fp32 [slots, heads, head_dim, head_dim])"
    return (f"{op}: no kernel for heads={state_cache.shape[1]}, head_dim={state_cache.shape[3]} "
            f"(covered: {sorted(_SHAPE_HOSTS)})")


def kda_prefill_shapes() -> frozenset[tuple[int, int]]:
    """The (heads, head_dim) with a kernel."""
    return frozenset(_SHAPE_HOSTS)


def kda_prefill_chunk_table(cu_seqlens, chunk_indices, state_indices, has_initial_state, *, state_cache):
    """The per-step chunk table ``kda_prefill`` takes as ``table`` (built once per step, shared by the layers)."""
    pkg = _shape_host(_key(state_cache))
    if pkg is None:
        raise ValueError(_no_kernel("kda_prefill_chunk_table", state_cache))
    return pkg.chunk_table(cu_seqlens, chunk_indices, state_indices, has_initial_state)


def kda_prefill(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices,
                has_initial_state, cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound,
                scale=None, table=None, chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None,
                checkpoint_rows=None, impl=None):
    """The KDA prefill of one layer (the routed package's ``prefill``); ``scale`` None: the package's default."""
    pkg = _shape_host(_key(state_cache))
    if pkg is None:
        raise ValueError(_no_kernel(_OP, state_cache))
    kw = {} if scale is None else {"scale": scale}
    return pkg.prefill(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache,
                       state_indices, has_initial_state, cu_seqlens, chunk_indices, chunk_offsets, norm_weight,
                       norm_eps, out, lower_bound, table=table, chunk_counts=chunk_counts,
                       checkpoint_state=checkpoint_state, checkpoint_offsets=checkpoint_offsets,
                       checkpoint_rows=checkpoint_rows, impl=impl, **kw)


def _layout(mixed, raw_g, beta_logit, gate, out, conv_state, h, d):
    bf = torch.bfloat16
    t = mixed.shape[0] if mixed.dim() == 2 else -1
    if mixed.dim() != 2 or mixed.dtype != bf or mixed.shape[1] != 3 * h * d or mixed.stride(1) != 1 or (
            mixed.stride(0) % 8):
        return f"mixed {tuple(mixed.shape)} {mixed.dtype} stride {mixed.stride()} (bf16 [T, {3 * h * d}], stride % 8)"
    if t < 1:
        return f"{t} tokens (the kernel takes at least 1)"
    if raw_g.dtype != bf or tuple(raw_g.shape) not in ((t, h, d), (t, h * d)) or raw_g.stride(-1) != 1 or (
            raw_g.dim() == 3 and raw_g.stride(1) != d):
        return f"raw_g {tuple(raw_g.shape)} {raw_g.dtype} stride {raw_g.stride()} (bf16 [{t}, {h}, {d}])"
    if beta_logit.dtype != bf or tuple(beta_logit.shape) != (t, h) or beta_logit.stride(1) != 1:
        return f"beta_logit {tuple(beta_logit.shape)} {beta_logit.dtype} (bf16 [{t}, {h}])"
    for name, x in (("gate", gate), ("out", out)):
        if x.dtype != bf or tuple(x.shape) != (t, h, d) or x.stride(1) != d or x.stride(2) != 1:
            return f"{name} {tuple(x.shape)} {x.dtype} stride {x.stride()} (bf16 [{t}, {h}, {d}], rows of {h * d})"
    cs = conv_state
    if cs.dim() != 3 or cs.dtype != bf or cs.shape[1] != 3 * h * d or cs.stride(0) >= 2**31 // 2:
        return f"conv state {tuple(cs.shape)} {cs.dtype} stride {cs.stride()} (bf16 [slots, {3 * h * d}, len])"
    return None


def kda_prefill_unsupported_reason(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache,
                                   state_indices, has_initial_state, cu_seqlens, chunk_indices, chunk_offsets,
                                   norm_weight, norm_eps, out, lower_bound, scale=None, table=None, chunk_counts=None,
                                   checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None,
                                   impl=None) -> str | None:
    """None when ``kda_prefill`` serves this call, else why not (host checks only)."""
    if not _dev.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(state_cache)
    if key not in _SHAPE_HOSTS:
        return _no_kernel(_OP, state_cache)
    h, d = key
    if state_cache.dtype != torch.float32 or state_cache.shape[2] != d or state_cache.stride(1) != d * d:
        return f"{_OP}: state cache {tuple(state_cache.shape)} {state_cache.dtype} (fp32 [slots, {h}, {d}, {d}])"
    reason = _layout(mixed, raw_g, beta_logit, gate, out, conv_state, h, d)
    if reason is not None:
        return f"{_OP}: {reason}"
    if (A_log.numel() != h or dt_bias.numel() != h * d or tuple(conv_weight.shape) != (3, 4, h * d)
            or not A_log.dtype == dt_bias.dtype == conv_weight.dtype == torch.float32
            or not (A_log.is_contiguous() and dt_bias.is_contiguous() and conv_weight.is_contiguous())):
        return (f"{_OP}: A_log {tuple(A_log.shape)} {A_log.dtype}, dt_bias {tuple(dt_bias.shape)} {dt_bias.dtype}, "
                f"conv weight {tuple(conv_weight.shape)} {conv_weight.dtype} (contiguous fp32 [{h}], [{h * d}], "
                f"[3, 4, {h * d}])")
    if norm_weight.shape != (d,) or norm_weight.dtype not in (torch.bfloat16, torch.float32):
        return f"{_OP}: norm weight {tuple(norm_weight.shape)} {norm_weight.dtype} (bf16 or fp32 [{d}])"
    n = cu_seqlens.numel() - 1
    if cu_seqlens.dtype != torch.int32 or cu_seqlens.dim() != 1 or n < 1:
        return f"{_OP}: cu_seqlens {tuple(cu_seqlens.shape)} {cu_seqlens.dtype} (int32 [sequences + 1])"
    if state_indices.dtype != torch.int32 or has_initial_state.dtype != torch.bool or state_indices.dim() != 1 or (
            has_initial_state.dim() != 1 or state_indices.numel() < n or has_initial_state.numel() < n):
        return (f"{_OP}: state indices {tuple(state_indices.shape)} {state_indices.dtype}, has_initial_state "
                f"{tuple(has_initial_state.shape)} {has_initial_state.dtype} (int32 [{n}], bool [{n}])")
    if chunk_indices.dtype != torch.int32 or chunk_indices.dim() != 2 or chunk_indices.shape[1] != 2 or (
            chunk_offsets.dtype != torch.int64):
        return f"{_OP}: chunk indices {tuple(chunk_indices.shape)} {chunk_indices.dtype}, offsets {chunk_offsets.dtype}"
    nc = chunk_indices.shape[0]
    if table is not None and (table.dtype != torch.int32 or tuple(table.shape) != (nc, 4) or not table.is_contiguous()):
        return f"{_OP}: table {tuple(table.shape)} {table.dtype} (int32 [{nc}, 4], kda_prefill_chunk_table)"
    if chunk_counts is not None and len(chunk_counts) != n:
        return f"{_OP}: {len(chunk_counts)} chunk counts for {n} sequences"
    if checkpoint_state is not None:
        ck = checkpoint_state
        if ck.dtype != torch.float32 or ck.dim() != 4 or tuple(ck.shape[1:]) != (h, d, d):
            return f"{_OP}: checkpoint state {tuple(ck.shape)} {ck.dtype} (fp32 [rows, {h}, {d}, {d}])"
        if checkpoint_offsets is None or checkpoint_offsets.dtype != torch.int32 or (
                checkpoint_rows is not None and checkpoint_rows.dtype != torch.int32):
            return f"{_OP}: checkpoint offsets / rows (int32)"
    if impl not in (None, "mega", "v1"):
        return f"{_OP}: impl {impl!r} (None, 'mega' or 'v1')"
    return None
