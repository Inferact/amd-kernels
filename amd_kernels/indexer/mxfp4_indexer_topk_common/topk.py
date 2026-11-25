# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP4 top-512 indexer of decode rows: routes a call by (index heads, head dim) of ``q`` to the shape package
(``_ROUTES``)."""

from amd_kernels.routes import Prepared, Routes, Rows, first, renamed

_ROUTES = Routes("mxfp4_indexer_topk", {(32, 128): "amd_kernels.indexer.indexer_mxfp4_h32_d128_top512"},
                 ("heads", "head_dim"))
MXFP4_INDEXER_MAX_ENTRIES = 1 << 20  # the entries' bound on a row's context (the long path's)
# decode rows at which the kernel beat AITER's indexer on MI355X (vLLM DeepSeek-V4.1 DEP8 routes)
_ROWS = Rows((1, 768))


def mxfp4_indexer_shapes() -> frozenset:
    """(index heads, head dim) keys."""
    return _ROUTES.shapes()


def _static(heads: int, head_dim: int, page_entries: int, max_ctx: int, next_n: int) -> str | None:
    key = (heads, head_dim)
    return first(
        _ROUTES.blocker(key),
        None if max_ctx <= MXFP4_INDEXER_MAX_ENTRIES else f"mxfp4_indexer_topk: max_ctx {max_ctx} above "
        f"{MXFP4_INDEXER_MAX_ENTRIES} entries",
    ) or _ROUTES.package(key).blocker(page_entries, max_ctx, next_n)


def _call(q, kv_cache, next_n, max_ctx) -> str | None:
    if q.dim() != 3 or kv_cache.dim() < 2:
        return "mxfp4_indexer_topk: q must be [rows, heads, head_dim / 2] packed MXFP4, kv_cache [pages, entries, ...]"
    return first(_static(q.shape[1], 2 * q.shape[2], kv_cache.shape[1], max_ctx, next_n),
                 _ROWS.blocker("mxfp4_indexer_topk", q.shape[0]))


def mxfp4_indexer_topk_blocker(q, q_scale, weights, kv_cache, block_table, row_ends, out, *, next_n, max_ctx,
                               **kwargs) -> str | None:  # fmt: skip
    """None when a package serves this call (contexts up to ``max_ctx`` entries; longer rows are truncated, so pass
    the true bound) and beats the served indexer at its row count."""
    return _call(q, kv_cache, next_n, max_ctx)


def mxfp4_indexer_candidate_topk_blocker(q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, *,
                                         next_n, max_ctx, **kwargs) -> str | None:  # fmt: skip
    """``mxfp4_indexer_topk_blocker`` for ``mxfp4_indexer_candidate_topk``."""
    return _call(q, kv_cache, next_n, max_ctx)


def _package(q):
    if q.dim() != 3:
        raise ValueError("mxfp4_indexer_topk: q must be [rows, heads, head_dim / 2] packed MXFP4")
    return _ROUTES.package((q.shape[1], 2 * q.shape[2]))


def mxfp4_indexer_topk(q, q_scale, weights, kv_cache, block_table, row_ends, out, *, next_n, max_ctx, **kwargs):
    """Top-512 keys of each decode row over the dense context (optionally the candidate list and the ragged slot
    lists); as the shape package's ``indexer_topk``."""
    return _package(q).indexer_topk(q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n=next_n,
                                    max_ctx=max_ctx, **kwargs)


def mxfp4_indexer_candidate_topk(q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, *, next_n,
                                 max_ctx, **kwargs):  # fmt: skip
    """Top-512 keys of each decode row over a candidate list; as the shape package's ``indexer_candidate_topk``."""
    return _package(q).indexer_candidate_topk(q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out,
                                              next_n=next_n, max_ctx=max_ctx, **kwargs)


def mxfp4_indexer_ragged(heads: int = 32, head_dim: int = 128, **fields):
    """The ragged slot-list request (``ragged=``) of the shape package (its ``Ragged`` fields)."""
    return _ROUTES.package((heads, head_dim)).Ragged(**fields)


def mxfp4_indexer_scores_cols(max_ctx: int, heads: int = 32, head_dim: int = 128) -> int | None:
    """Columns of the fp32 block-maxima workspace [rows, cols] (``scores=``) a caller may share across layers for
    ``max_ctx``; None on the token path, where the package keeps its own."""
    host = _ROUTES.package((heads, head_dim)).host
    if host.tokens_tier(max_ctx) is not None:
        return None
    return host.blocks_tier(max_ctx) + 64


class Mxfp4IndexerTopk(Prepared):
    """One indexer layer's top-512 bound to its static bounds: ``h(q, q_scale, weights, kv_cache, block_table,
    row_ends, out, **kwargs)``, ``h.candidate_topk(..., candidates, out, **kwargs)``."""

    op = "mxfp4_indexer_topk"

    def __init__(self, next_n: int, max_ctx: int):
        self.next_n, self.max_ctx = next_n, max_ctx

    def blocker(self, q, q_scale=None, weights=None, kv_cache=None, *args, **kwargs) -> str | None:
        return _ROWS.blocker(self.op, q.shape[0])

    def __call__(self, q, q_scale, weights, kv_cache, block_table, row_ends, out, **kwargs):
        return mxfp4_indexer_topk(q, q_scale, weights, kv_cache, block_table, row_ends, out, next_n=self.next_n,
                                  max_ctx=self.max_ctx, **kwargs)

    def candidate_topk(self, q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out, **kwargs):
        return mxfp4_indexer_candidate_topk(q, q_scale, weights, kv_cache, block_table, row_ends, candidates, out,
                                            next_n=self.next_n, max_ctx=self.max_ctx, **kwargs)


def prepare_mxfp4_indexer_topk(heads: int, head_dim: int, page_entries: int, max_ctx: int,
                               next_n: int) -> Mxfp4IndexerTopk | str:  # fmt: skip
    """The layer's indexer, or the reason no package serves ``heads`` x ``head_dim`` MXFP4 queries over pages of
    ``page_entries`` keys with contexts up to ``max_ctx`` entries (a load-time check: per call only the row count
    is left to decide)."""
    reason = _static(heads, head_dim, page_entries, max_ctx, next_n)
    return reason if reason is not None else Mxfp4IndexerTopk(next_n, max_ctx)


mxfp4_indexer_blocker = renamed(_static, "mxfp4_indexer_blocker", "prepare_mxfp4_indexer_topk")
