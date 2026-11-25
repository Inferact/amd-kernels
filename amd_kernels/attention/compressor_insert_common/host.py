# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compressor pool + RMSNorm + RoPE + latent insert and the indexer key store: routes a call by (latent dim, index key
dim, rotary dim) of the main cache, ``wk`` and the main cos / sin cache to the shape package (``_ROUTES``)."""

from amd_kernels.routes import Routes, arity, check, previous_form

_ROUTES = Routes("compressor_insert", {(512, 128, 64): "amd_kernels.attention.compressor_insert_d512_idx128_rot64"},
                 ("dim", "index_dim", "rotary_dim"))


def _key(main_cache, wk, cos_sin_main):
    if main_cache.dim() != 3 or wk.dim() != 2 or cos_sin_main.dim() != 2:
        return None
    return (main_cache.shape[2], wk.shape[0], cos_sin_main.shape[1])


def compressor_insert_shapes() -> frozenset:
    """(latent dim, index key dim, rotary dim) keys."""
    return _ROUTES.shapes()


def _unsupported(raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx) -> str | None:
    key = _key(main_cache, wk, cos_sin_main)
    if key is None:
        return "compressor_insert: main_cache must be [pages, rows, dim], wk [index dim, dim], cos_sin [positions, rot]"
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not _ROUTES.package(key).supports_compressor_insert(raw, ratio, main_cache, idx_cache, wk, cos_sin_main,
                                                           cos_sin_idx):
        return "compressor_insert: the kernel does not take these inputs (dtype, layout, ratio or device)"
    return None


@previous_form(_unsupported, arity(7))
def compressor_insert_blocker(raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps,
                              main_cache, main_slots, cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache,
                              idx_slots, cos_sin_idx, ratio) -> str | None:  # fmt: skip
    """None when a package serves the compressor tail + indexer key store of these tensors (``ratio`` 1 or 2)."""
    return _unsupported(raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx)


def compressor_shuffle_wk(wk):
    """``wk`` [index dim, latent dim] bf16 in the layout ``compressor_insert`` reads (load time)."""
    keys = [k for k in _ROUTES.hosts if k[:2] == (wk.shape[1], wk.shape[0])]
    check(None if keys else f"compressor_insert: no kernel for wk {tuple(wk.shape)}")
    return _ROUTES.package(keys[0]).shuffle_wk(wk)


def compressor_insert(raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps, main_cache,
                      main_slots, cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache, idx_slots, cos_sin_idx,
                      ratio):  # fmt: skip
    """Compressor pool + RMSNorm + RoPE + latent insert and the indexer key store in one launch; as the shape
    package's ``compressor_insert``."""
    check(_unsupported(raw, ratio, main_cache, idx_cache, wk, cos_sin_main, cos_sin_idx))
    return _ROUTES.package(_key(main_cache, wk, cos_sin_main)).compressor_insert(
        raw, positions, ring, ring_slots, query_start_loc, token_to_req, norm_weight, eps, main_cache, main_slots,
        cos_sin_main, kv_scale, wk, k_norm_weight, k_eps, idx_cache, idx_slots, cos_sin_idx, ratio,
    )  # fmt: skip
