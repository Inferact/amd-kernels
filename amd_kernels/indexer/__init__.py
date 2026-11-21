# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexer op entries (``amd_kernels.routes``): the MXFP4 top-512 indexer and the indexer query RoPE + MXFP4 quant."""

from amd_kernels.routes import Routes, check, first

_MXFP4_INDEXER = Routes("mxfp4_indexer_topk", {(32, 128): "amd_kernels.indexer.indexer_mxfp4_h32_d128_top512"})
_Q_ROPE = Routes("q_rope_mxfp4_quant", {(32, 128, 64): "amd_kernels.indexer.q_rope_mxfp4_quant_h32_d128_rot64"})
MXFP4_INDEXER_MAX_ENTRIES = 1 << 20  # the entries' bound on a row's context (the long path's)


def mxfp4_indexer_shapes() -> frozenset:
    """(index heads, head dim) keys."""
    return _MXFP4_INDEXER.shapes()


def mxfp4_indexer_blocker(heads: int, head_dim: int, page_entries: int, max_ctx: int, next_n: int) -> str | None:
    """None when a package serves the top-512 of decode rows of ``heads`` x ``head_dim`` MXFP4 queries over pages of
    ``page_entries`` keys, contexts up to ``max_ctx`` entries (longer rows are truncated: pass the true bound)."""
    key = (heads, head_dim)
    return first(
        _MXFP4_INDEXER.blocker(key),
        None if max_ctx <= MXFP4_INDEXER_MAX_ENTRIES else f"mxfp4_indexer_topk: max_ctx {max_ctx} above "
        f"{MXFP4_INDEXER_MAX_ENTRIES} entries",
    ) or _MXFP4_INDEXER.package(key).blocker(page_entries, max_ctx, next_n)


def _indexer_package(q):
    if q.dim() != 3:
        raise ValueError("mxfp4_indexer_topk: q must be [rows, heads, head_dim / 2] packed MXFP4")
    return _MXFP4_INDEXER.package((q.shape[1], 2 * q.shape[2]))


def mxfp4_indexer_topk(q, q_scale, weights, kv_cache, block_table, row_ends, out, *, next_n, max_ctx, **kwargs):
    """Top-512 keys of each decode row over the dense context (optionally the candidate list and the ragged slot
    lists); as the shape package's ``indexer_topk``. The caller checks ``mxfp4_indexer_blocker`` once per layer."""
    return _indexer_package(q).indexer_topk(
        q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n=next_n, max_ctx=max_ctx, **kwargs
    )


def mxfp4_indexer_candidate_topk(q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, *, next_n,
                                 max_ctx, **kwargs):  # fmt: skip
    """Top-512 keys of each decode row over a candidate list; as the shape package's ``indexer_candidate_topk``."""
    return _indexer_package(q).indexer_candidate_topk(
        q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, next_n=next_n, max_ctx=max_ctx,
        **kwargs,
    )  # fmt: skip


def mxfp4_indexer_ragged(heads: int = 32, head_dim: int = 128, **fields):
    """The ragged slot-list request (``ragged=``) of the shape package (its ``Ragged`` fields)."""
    return _MXFP4_INDEXER.package((heads, head_dim)).Ragged(**fields)


def mxfp4_indexer_scores_cols(max_ctx: int, heads: int = 32, head_dim: int = 128) -> int | None:
    """Columns of the fp32 block-maxima workspace [rows, cols] (``scores=``) a caller may share across layers for
    ``max_ctx``; None on the token path, where the package keeps its own."""
    host = _MXFP4_INDEXER.package((heads, head_dim)).host
    if host.tokens_tier(max_ctx) is not None:
        return None
    return host.blocks_tier(max_ctx) + 64


def _q_rope_key(q, cos_sin_cache):
    return (q.shape[1], q.shape[2], cos_sin_cache.shape[1]) if q.dim() == 3 and cos_sin_cache.dim() == 2 else None


def q_rope_mxfp4_quant_shapes() -> frozenset:
    """(heads, head dim, rotary dim) keys."""
    return _Q_ROPE.shapes()


def q_rope_mxfp4_quant_blocker(q, cos_sin_cache) -> str | None:
    """None when a package serves the indexer query RoPE + MXFP4 quant of ``q`` [rows, heads, dim] bf16."""
    key = _q_rope_key(q, cos_sin_cache)
    if key is None:
        return "q_rope_mxfp4_quant: q must be [rows, heads, dim], cos_sin_cache [positions, rot]"
    reason = _Q_ROPE.blocker(key)
    if reason is not None:
        return reason
    if not _Q_ROPE.package(key).supports_q_rope_mxfp4_quant(q, cos_sin_cache):
        return "q_rope_mxfp4_quant: the kernel does not take these inputs (dtype, layout or device)"
    return None


def q_rope_mxfp4_quant(q, positions, cos_sin_cache, *args, **kwargs):
    """Indexer query RoPE + MXFP4 quant + weight scaling in one launch; as the shape package's
    ``q_rope_mxfp4_quant`` (returns codes, scales, scaled weights)."""
    check(q_rope_mxfp4_quant_blocker(q, cos_sin_cache))
    return _Q_ROPE.package(_q_rope_key(q, cos_sin_cache)).q_rope_mxfp4_quant(q, positions, cos_sin_cache, *args,
                                                                             **kwargs)
