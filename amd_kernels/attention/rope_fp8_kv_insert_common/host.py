# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Query RoPE + e4m3 quant and KV RoPE + e4m3 cache insert: routes a call by (heads, head dim, rotary dim) of ``q``
and ``cos_sin_cache`` to the shape package (``_ROUTES``)."""

from amd_kernels.routes import Routes, arity, check, previous_form

_ROUTES = Routes("rope_fp8_kv_insert", {(64, 512, 64): "amd_kernels.attention.rope_fp8_kv_insert_h64_d512_rot64"},
                 ("heads", "dim", "rotary_dim"))


def _key(q, cos_sin_cache):
    return (q.shape[1], q.shape[2], cos_sin_cache.shape[1]) if q.dim() == 3 and cos_sin_cache.dim() == 2 else None


def rope_fp8_kv_insert_shapes() -> frozenset:
    """(heads, head dim, rotary dim) keys."""
    return _ROUTES.shapes()


def _unsupported(q, kv_cache, cos_sin_cache) -> str | None:
    key = _key(q, cos_sin_cache)
    if key is None:
        return "rope_fp8_kv_insert: q must be [rows, heads, dim], cos_sin_cache [positions, rot]"
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not _ROUTES.package(key).supports_rope_fp8_kv_insert(q, kv_cache, cos_sin_cache):
        return "rope_fp8_kv_insert: the kernel does not take these inputs (dtype, layout or device)"
    return None


@previous_form(_unsupported, arity(3))
def rope_fp8_kv_insert_blocker(q, kv, q_fp8, kv_cache, slot_mapping, positions, cos_sin_cache, kv_scale=None,
                               q_scale_inv=None, n_insert=None) -> str | None:  # fmt: skip
    """None when a package serves the query RoPE + e4m3 quant and the KV RoPE + e4m3 insert of these tensors."""
    return _unsupported(q, kv_cache, cos_sin_cache)


def rope_fp8_kv_insert(q, kv, q_fp8, kv_cache, slot_mapping, positions, cos_sin_cache, kv_scale, q_scale_inv,
                       n_insert=None):  # fmt: skip
    """``q_fp8`` = e4m3(RoPE(q) * q_scale_inv); RoPE(kv) of the rows with a slot into the e4m3 cache; as the shape
    package's ``rope_fp8_kv_insert``."""
    check(_unsupported(q, kv_cache, cos_sin_cache))
    return _ROUTES.package(_key(q, cos_sin_cache)).rope_fp8_kv_insert(
        q, kv, q_fp8, kv_cache, slot_mapping, positions, cos_sin_cache, kv_scale, q_scale_inv, n_insert)
