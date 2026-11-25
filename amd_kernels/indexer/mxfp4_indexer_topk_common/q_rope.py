# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The indexer query RoPE + MXFP4 quant + weight scaling: routes a call by (heads, head dim, rotary dim) to the shape
package (``_ROUTES``)."""

from amd_kernels.routes import Routes, check

_ROUTES = Routes("q_rope_mxfp4_quant", {(32, 128, 64): "amd_kernels.indexer.q_rope_mxfp4_quant_h32_d128_rot64"},
                 ("heads", "head_dim", "rotary_dim"))


def _key(q, cos_sin_cache):
    return (q.shape[1], q.shape[2], cos_sin_cache.shape[1]) if q.dim() == 3 and cos_sin_cache.dim() == 2 else None


def q_rope_mxfp4_quant_shapes() -> frozenset:
    """(heads, head dim, rotary dim) keys."""
    return _ROUTES.shapes()


def q_rope_mxfp4_quant_blocker(q, positions=None, cos_sin_cache=None, *args, **kwargs) -> str | None:
    """None when a package serves the indexer query RoPE + MXFP4 quant of ``q`` [rows, heads, dim] bf16."""
    if cos_sin_cache is None:  # the previous form: (q, cos_sin_cache)
        positions, cos_sin_cache = None, positions
    key = _key(q, cos_sin_cache)
    if key is None:
        return "q_rope_mxfp4_quant: q must be [rows, heads, dim], cos_sin_cache [positions, rot]"
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not _ROUTES.package(key).supports_q_rope_mxfp4_quant(q, cos_sin_cache):
        return "q_rope_mxfp4_quant: the kernel does not take these inputs (dtype, layout or device)"
    return None


def q_rope_mxfp4_quant(q, positions, cos_sin_cache, *args, **kwargs):
    """Indexer query RoPE + MXFP4 quant + weight scaling in one launch; as the shape package's
    ``q_rope_mxfp4_quant`` (returns codes, scales, scaled weights)."""
    check(q_rope_mxfp4_quant_blocker(q, positions, cos_sin_cache))
    return _ROUTES.package(_key(q, cos_sin_cache)).q_rope_mxfp4_quant(q, positions, cos_sin_cache, *args, **kwargs)
