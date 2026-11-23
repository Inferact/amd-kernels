# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prefill top-k of a lightning indexer over NVFP4 index pages (scorer + top-k of every prefill row in one call):
routes by (index_head_dim, page, top-k), read from iq [rows, heads, d], the index cache [pages, page, 72] and topk_idx
[heads, rows, k], to the shape package (``_ROUTES``, imported on the first call of its key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes

_NAMES = ("index_head_dim", "page", "topk")
_HOSTS = {(128, 128, 16): "amd_kernels.indexer.indexer_nvfp4_d128_p128_top16"}
_ROUTES = Routes("index_prefill_top16", _HOSTS, _NAMES)
_LAYER = Routes("index_prefill_top16_layer", _HOSTS, _NAMES)
_OP = _ROUTES.op
_I32 = torch.int32


def _key(iq, index_cache, topk_idx):
    if iq.dim() != 3 or index_cache.dim() != 3 or topk_idx.dim() != 3:
        return None
    return (iq.shape[2], index_cache.shape[1], topk_idx.shape[2])


def _module(key):
    return _ROUTES.module(key, "prefill_top16")


def index_prefill_top16_shapes() -> frozenset[tuple[int, int, int]]:
    """The (index_head_dim, page, top-k) with a kernel."""
    return _ROUTES.shapes()


def index_prefill_top16_layer_blocker(*, num_index_heads: int, index_head_dim: int, block_size: int,
                                      topk_blocks: int) -> str | None:
    """None when a model's prefill indexer (index heads per GPU, head dim, block size, top-k) fits this family: a
    load-time check."""
    key = (index_head_dim, block_size, topk_blocks)
    reason = _LAYER.blocker(key)
    if reason is not None:
        return reason
    reason = _LAYER.module(key, "prefill_top16").index_prefill_top16_blocker(
        num_index_heads=num_index_heads, index_head_dim=index_head_dim, block_size=block_size,
        topk_blocks=topk_blocks)
    return None if reason is None else f"{_LAYER.op}: {reason}"


def index_prefill_top16(iq, index_cache, block_table, cu_seqlens_q, seq_lens, init_blocks, local_blocks, topk_idx, *,
                        max_query_len, max_seq_len, num_valid_pages, score=None, stream=None):
    """The top-k block ids of every prefill row into ``topk_idx`` (the routed package's ``index_prefill_top16``)."""
    return _module(_key(iq, index_cache, topk_idx)).index_prefill_top16(
        iq, index_cache, block_table, cu_seqlens_q, seq_lens, init_blocks, local_blocks, topk_idx,
        max_query_len=max_query_len, max_seq_len=max_seq_len, num_valid_pages=num_valid_pages, score=score,
        stream=stream)


def index_prefill_top16_blocker(iq, index_cache, block_table, cu_seqlens_q, seq_lens, init_blocks, local_blocks,
                                topk_idx, *, max_query_len, max_seq_len, num_valid_pages, score=None,
                                stream=None) -> str | None:
    """None when ``index_prefill_top16`` serves this call, else why not (host checks only)."""
    if not routes.available():
        return f"{_OP}: needs gfx950 with FlyDSL"
    key = _key(iq, index_cache, topk_idx)
    if key is None:
        return (f"{_OP}: iq {tuple(iq.shape)} ([rows, heads, d]), index cache {tuple(index_cache.shape)} ([pages, "
                f"page, 72]), top-k {tuple(topk_idx.shape)} ([heads, rows, k])")
    reason = _ROUTES.blocker(key)
    if reason is not None:
        return reason
    mod = _module(key)
    reason = mod.index_prefill_top16_blocker(num_index_heads=iq.shape[1], index_head_dim=key[0], block_size=key[1],
                                             topk_blocks=key[2])
    if reason is not None:
        return f"{_OP}: {reason}"
    rows = iq.shape[0]
    if iq.dtype != torch.float8_e4m3fn or iq.stride(2) != 1 or iq.stride(1) != key[0]:
        return f"{_OP}: iq {iq.dtype} stride {iq.stride()} (e4m3, rows contiguous)"
    if index_cache.dtype != torch.uint8 or index_cache.shape[2] != 72:
        return f"{_OP}: index cache {tuple(index_cache.shape)} {index_cache.dtype} (uint8 [pages, {key[1]}, 72])"
    if topk_idx.dtype != _I32 or tuple(topk_idx.shape[:2]) != (iq.shape[1], rows):
        return f"{_OP}: top-k {tuple(topk_idx.shape)} {topk_idx.dtype} (int32 [{iq.shape[1]}, {rows}, {key[2]}])"
    for name, t in (("block table", block_table), ("cu_seqlens_q", cu_seqlens_q), ("seq lens", seq_lens)):
        if t.dtype != _I32 or t.stride(-1) != 1:
            return f"{_OP}: {name} {t.dtype} (int32, unit last stride)"
    if num_valid_pages.dtype != _I32 or num_valid_pages.shape[0] < rows:
        return f"{_OP}: num_valid_pages {tuple(num_valid_pages.shape)} {num_valid_pages.dtype} (int32 [>= {rows}])"
    if not 1 <= max_seq_len <= mod.MAX_SEQ_LEN:
        return f"{_OP}: max_seq_len {max_seq_len} (1..{mod.MAX_SEQ_LEN})"
    return None


def warm_up_index_prefill_top16(device, *, num_index_heads: int, index_head_dim: int, block_size: int,
                                topk_blocks: int) -> None:
    """Compile every variant the prefill top-k launches for this many index heads (before serving)."""
    _LAYER.module((index_head_dim, block_size, topk_blocks), "prefill_top16").warm_up(device, num_index_heads)
