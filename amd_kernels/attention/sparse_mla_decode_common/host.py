# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode with the inverse-RoPE MXFP8 epilogue: routes a call by (query heads, head dim) of ``q`` to the
shape package (``_ROUTES``)."""

from amd_kernels.routes import Routes, Rows, check, first, previous_form

_ROUTES = Routes("sparse_mla_decode", {(64, 512): "amd_kernels.attention.sparse_mla_decode_fp8kv_h64_d512_rot64"},
                 ("heads", "dim"))
# decode rows at which the kernel beat AITER's sparse MLA + MXFP8 quant on MI355X (vLLM DeepSeek-V4.1 DEP8 routes)
_ROWS = Rows((1, 768))


def sparse_mla_decode_shapes() -> frozenset:
    """(query heads, head dim) keys."""
    return _ROUTES.shapes()


def _unsupported(q, main_cache, num_heads=None, keys_per_row=None) -> str | None:
    if q.dim() != 3:
        return "sparse_mla_decode: q must be [rows, heads, dim]"
    key = tuple(q.shape[1:])
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    if not _ROUTES.package(key).supports_sparse_mla_decode(q, main_cache, num_heads, keys_per_row):
        return (f"sparse_mla_decode: {_ROUTES.hosts[key].rsplit('.', 1)[-1]} does not take these inputs (dtype, "
                "layout, device or more keys per row than it holds)")
    return None


def _ints(args, kwargs) -> bool:
    return 2 <= len(args) <= 4 and all(a is None or type(a) is int for a in args[2:]) and set(kwargs) <= {
        "num_heads", "keys_per_row"}


@previous_form(_unsupported, _ints)
def sparse_mla_decode_blocker(q, main_cache, main_indptr=None, main_indices=None, extra_cache=None,
                              extra_indptr=None, extra_indices=None, attn_sink=None, softmax_scale=None,
                              q_scale=None, kv_scale=None, positions=None, cos_sin=None, out_codes=None,
                              out_scales=None, out_bf16=None, *, keys_per_row: int | None = None,
                              **kwargs) -> str | None:  # fmt: skip
    """None when a package serves this call and beats the served path at its row count. ``keys_per_row``: the most
    keys a row has (window plus top-k), which the tensors do not show."""
    return first(_unsupported(q, main_cache, None, keys_per_row), _ROWS.blocker("sparse_mla_decode", q.shape[0]))


def sparse_mla_decode(q, main_cache, main_indptr, main_indices, extra_cache, extra_indptr, extra_indices, attn_sink,
                      softmax_scale, q_scale, kv_scale, positions, cos_sin, out_codes, out_scales, out_bf16=None,
                      **kwargs):  # fmt: skip
    """Attention of every query row over its main (window) and extra (top-k) key lists of an e4m3 cache, then the
    inverse RoPE and the MXFP8 quant into ``out_codes`` / ``out_scales`` (or bf16 ``out_bf16``); as the shape
    package's ``sparse_mla_decode``."""
    check(_unsupported(q, main_cache))
    return _ROUTES.package(tuple(q.shape[1:])).sparse_mla_decode(
        q, main_cache, main_indptr, main_indices, extra_cache, extra_indptr, extra_indices, attn_sink, softmax_scale,
        q_scale, kv_scale, positions, cos_sin, out_codes, out_scales, out_bf16, **kwargs)
