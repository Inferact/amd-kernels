# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Dense fp8 MLA decode over a context-parallel round-robin shard: key (heads, d_qk) from ``q``. ``cp_world`` is a
run-time argument the kernels compile for (any value), not a key dim."""

import functools
import importlib

import torch

from .routing import available, no_kernel

_SHAPE_HOSTS = {
    (96, 576): "amd_kernels.attention.fp8_dense_mla_decode_h96_kv512.host",
}
_KEY = ("heads", "d_qk")
_OP = "fp8_dense_mla_decode"


@functools.cache
def _shape_host(key):
    name = _SHAPE_HOSTS.get(key)
    return importlib.import_module(name) if name is not None else None


def fp8_dense_mla_decode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, sm_scale, query_len,
                         cp_rank=0, cp_world=8, out=None, lse=None):
    """``(out, lse)``: the partial latent attention output [rows, heads, kv_lora] bf16 and the natural-log LSE
    [rows, heads] fp32 of ``q`` [requests * query_len, heads, d_qk] e4m3 over this rank's shard of the context."""
    key = (q.shape[1], q.shape[2])
    pkg = _shape_host(key)
    if pkg is None:
        raise ValueError(no_kernel(_OP, _KEY, key, _SHAPE_HOSTS))
    return pkg.fp8_dense_mla_decode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale, sm_scale,
                                    query_len, cp_rank, cp_world, out, lse)


def fp8_dense_mla_decode_shapes() -> frozenset:
    return frozenset(_SHAPE_HOSTS)


def fp8_dense_mla_decode_unsupported_reason(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale,
                                            sm_scale, query_len, cp_rank=0, cp_world=8, out=None,
                                            lse=None) -> str | None:
    op = _OP
    if q.dim() != 3:
        return f"{op}: operand layout (q [rows, heads, d_qk])"
    key = (q.shape[1], q.shape[2])
    if key not in _SHAPE_HOSTS:
        return no_kernel(op, _KEY, key, _SHAPE_HOSTS)
    if not available():
        return f"{op}: needs gfx950 with FlyDSL"
    host = _shape_host(key)
    if not 1 <= query_len <= host.MAX_QLEN:
        return f"{op}: query_len {query_len} (the kernel takes 1..{host.MAX_QLEN})"
    rows = q.shape[0]
    if rows % query_len:
        return f"{op}: operand layout ({rows} rows, not requests * query_len {query_len})"
    if rows // query_len > host.MAX_REQS:
        return f"{op}: {rows // query_len} requests (the kernel takes 0..{host.MAX_REQS})"
    if not 0 <= cp_rank < cp_world:
        return f"{op}: cp_rank {cp_rank} of cp_world {cp_world}"
    if not host.supports_fp8_dense_mla_decode(q, kv_cache, kv_indptr, kv_indices, g_kv_indptr, q_scale, kv_scale,
                                              query_len, out):
        return f"{op}: operand layout (supports_fp8_dense_mla_decode)"
    if lse is not None and (lse.dtype != torch.float32 or lse.shape != (rows, key[0]) or lse.stride(1) != 1):
        return f"{op}: operand layout (lse fp32 [rows, {key[0]}], unit column stride)"
    return None
