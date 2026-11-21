# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Attention op entries (``amd_kernels.routes``): sparse MLA decode, RoPE + fp8 KV insert, compressor insert."""

from amd_kernels.routes import Routes, check, first

_SPARSE_MLA = Routes(
    "sparse_mla_decode", {(64, 512): "amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64"}
)
_ROPE_INSERT = Routes(
    "rope_fp8_kv_insert", {(64, 512, 64): "amd_kernels.attention.rope_fp8_kv_insert_h64_d512_rot64"}
)
_COMPRESSOR = Routes(
    "compressor_insert", {(512, 128, 64): "amd_kernels.attention.compressor_insert_d512_idx128_rot64"}
)


def _unsupported(op: str, routes: Routes, key) -> str:
    return f"{op}: {routes.hosts[key].rsplit('.', 1)[-1]} does not take these inputs (dtype, layout or device)"


def sparse_mla_decode_shapes() -> frozenset:
    """(query heads, head dim) keys."""
    return _SPARSE_MLA.shapes()


def sparse_mla_decode_blocker(q, main_cache, num_heads: int | None = None) -> str | None:
    """None when a package serves sparse MLA decode for ``q`` [rows, heads, dim] over ``main_cache``."""
    if q.dim() != 3:
        return "sparse_mla_decode: q must be [rows, heads, dim]"
    key = tuple(q.shape[1:])
    reason = _SPARSE_MLA.blocker(key)
    if reason is not None:
        return reason
    ok = _SPARSE_MLA.package(key).supports_sparse_mla_decode(q, main_cache, num_heads)
    return None if ok else _unsupported("sparse_mla_decode", _SPARSE_MLA, key)


def sparse_mla_decode(q, main_cache, *args, **kwargs):
    """Sparse MLA decode over an e4m3 KV cache with the inverse-RoPE MXFP8 epilogue; arguments and outputs as the
    shape package's ``sparse_mla_decode`` (``sparse_mla_decode_fp8kv_h64_d512_rot64``)."""
    check(sparse_mla_decode_blocker(q, main_cache))
    return _SPARSE_MLA.package(tuple(q.shape[1:])).sparse_mla_decode(q, main_cache, *args, **kwargs)


def _rope_key(q, cos_sin_cache):
    return (q.shape[1], q.shape[2], cos_sin_cache.shape[1]) if q.dim() == 3 and cos_sin_cache.dim() == 2 else None


def rope_fp8_kv_insert_shapes() -> frozenset:
    """(heads, head dim, rotary dim) keys."""
    return _ROPE_INSERT.shapes()


def rope_fp8_kv_insert_blocker(q, kv_cache, cos_sin_cache) -> str | None:
    """None when a package serves the query RoPE + e4m3 quant and the KV RoPE + e4m3 insert of these tensors."""
    key = _rope_key(q, cos_sin_cache)
    if key is None:
        return "rope_fp8_kv_insert: q must be [rows, heads, dim], cos_sin_cache [positions, rot]"
    reason = _ROPE_INSERT.blocker(key)
    if reason is not None:
        return reason
    ok = _ROPE_INSERT.package(key).supports_rope_fp8_kv_insert(q, kv_cache, cos_sin_cache)
    return None if ok else _unsupported("rope_fp8_kv_insert", _ROPE_INSERT, key)


def rope_fp8_kv_insert(q, kv, q_fp8, kv_cache, slot_mapping, positions, cos_sin_cache, *args, **kwargs):
    """Query RoPE + e4m3 quant and KV RoPE + e4m3 cache insert; as the shape package's ``rope_fp8_kv_insert``."""
    check(rope_fp8_kv_insert_blocker(q, kv_cache, cos_sin_cache))
    return _ROPE_INSERT.package(_rope_key(q, cos_sin_cache)).rope_fp8_kv_insert(
        q, kv, q_fp8, kv_cache, slot_mapping, positions, cos_sin_cache, *args, **kwargs
    )


def _compressor_key(main_cache, wk, cos_sin_main):
    if main_cache.dim() != 3 or wk.dim() != 2 or cos_sin_main.dim() != 2:
        return None
    return (main_cache.shape[2], wk.shape[0], cos_sin_main.shape[1])


def compressor_insert_shapes() -> frozenset:
    """(latent dim, index key dim, rotary dim) keys."""
    return _COMPRESSOR.shapes()


def compressor_insert_blocker(raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx) -> str | None:
    """None when a package serves the compressor tail + indexer key store of these tensors (``ratio`` 1 or 2)."""
    key = _compressor_key(main_cache, wk, cos_sin_main)
    if key is None:
        return "compressor_insert: main_cache must be [pages, rows, dim], wk [index dim, dim], cos_sin [positions, rot]"
    reason = _COMPRESSOR.blocker(key)
    if reason is not None:
        return reason
    ok = _COMPRESSOR.package(key).supports_compressor_insert(
        raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx
    )
    return None if ok else _unsupported("compressor_insert", _COMPRESSOR, key)


def compressor_shuffle_wk(wk):
    """``wk`` [index dim, latent dim] bf16 in the layout ``compressor_insert`` reads (load time)."""
    keys = [k for k in _COMPRESSOR.hosts if k[:2] == (wk.shape[1], wk.shape[0])]
    check(None if keys else f"compressor_insert: no shape package for wk {tuple(wk.shape)}")
    return _COMPRESSOR.package(keys[0]).shuffle_wk(wk)


def compressor_insert(raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps, main_cache,
                      main_slots, cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache, idx_slots, cos_sin_idx,
                      ratio):  # fmt: skip
    """Compressor pool + RMSNorm + RoPE + latent insert and the indexer key store in one launch; as the shape
    package's ``compressor_insert``."""
    check(compressor_insert_blocker(raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx))
    return _COMPRESSOR.package(_compressor_key(main_cache, wk, cos_sin_main)).compressor_insert(
        raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps, main_cache, main_slots,
        cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache, idx_slots, cos_sin_idx, ratio,
    )  # fmt: skip
