# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode over fp8_ds_mla records: key (heads, d_qk, record bytes) from ``q`` and ``kv_cache``; the
pre-quantized ``q8`` form matches on (heads, record bytes)."""

from amd_kernels import routes
from amd_kernels.routes import Routes, renamed

_ROUTES = Routes("fp8_sparse_mla_decode",
                 {(96, 576, 656): "amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048"},
                 ("heads", "d_qk", "record"))
_OP = _ROUTES.op


def _q8_key(heads, record):
    return next((k for k in _ROUTES.hosts if (k[0], k[2]) == (heads, record)), (heads, "q8", record))


def _key(q, kv_cache, q8):
    if q8 is not None:
        return _q8_key(q8.shape[1], kv_cache.shape[-1])
    return (q.shape[1], q.shape[2], kv_cache.shape[-1])


def _host(q, kv_cache, q8):
    return _ROUTES.package(_key(q, kv_cache, q8))


def fp8_sparse_mla_decode(q, kv_cache, indptr, slots, sm_scale, out=None, q8=None):
    """``(out, lse)``: the partial latent attention output [rows, heads, kv_lora] bf16 and the natural-log LSE
    [rows, heads] fp32 of ``q`` (or the pre-quantized ``q8``) over the CSR ``slots`` of the cache."""
    return _host(q, kv_cache, q8).fp8_sparse_mla_decode(q, kv_cache, indptr, slots, sm_scale, out, q8)


def fp8_sparse_mla_decode_shapes() -> frozenset:
    return _ROUTES.shapes()


def fp8_sparse_mla_decode_blocker(q, kv_cache, indptr, slots, sm_scale, out=None, q8=None) -> str | None:
    t = q8 if q8 is not None else q
    if t is None or t.dim() != 3 or kv_cache.dim() < 1:
        return f"{_OP}: operand layout (q or q8 [rows, heads, d], kv_cache [..., record])"
    key = _key(q, kv_cache, q8)
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    pkg = _ROUTES.package(key)
    cap = pkg.host.MAX_ROWS
    if t.shape[0] > cap:
        return f"{_OP}: {t.shape[0]} rows (the kernel takes 0..{cap})"
    if not pkg.supports_fp8_sparse_mla_decode(q, kv_cache, indptr, slots, out, q8):
        return f"{_OP}: operand layout (supports_fp8_sparse_mla_decode)"
    return None


def fp8_sparse_mla_decode_kernel(*args, **kwargs):
    """The routed ``amd_kernels.warm_up`` entry (``.entry``) for these arguments."""
    return _host(*_bind(*args, **kwargs)).fp8_sparse_mla_decode


def _bind(q, kv_cache, indptr, slots, sm_scale, out=None, q8=None):
    return q, kv_cache, q8


fp8_sparse_mla_decode_unsupported_reason = renamed(fp8_sparse_mla_decode_blocker,
                                                   "fp8_sparse_mla_decode_unsupported_reason")
