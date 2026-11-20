# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode block of one sparse layer in one call: the index top-16 of every decode row, then the sparse decode
attention over those blocks.

Routes (``route``; the ids and the attention output are bit for bit the same on every route):

- ``split4``: the four-GPU scorer + top-16 (``decode_top16.IndexDecodeTop16``, from ``decode_top16.SPLIT_MIN_REQS``
  requests), then the attention on its ids;
- ``split2``: the two-GPU scorer + top-16 (TP2, from ``decode_top16.SPLIT2_MIN_REQS`` requests), then the attention on
  its ids;
- ``fused`` (one GPU, <= ``host.SELECT_MAX_ROWS`` rows, one index head per KV head, bf16 or fp16): the scorer
  (``nvfp4_index_score_rows``), then the attention with ``index_score=``, whose workgroups select the top-16
  themselves: -0.84..-1.43 us per layer at 4-32 rows, -0.7..-2.2 us at 40-256 rows (1 KV head, 32K-192K, bf16), -1.0..
  -2.0 us at 8-1024 rows (2 KV heads, fp16) against ``chain``;
- ``chain`` (one GPU, otherwise): ``nvfp4_index_score_top16`` (scorer + selector), then the attention.

``topk_idx`` holds the ids afterwards on every route (a layer that reuses the previous layer's top-16 reads it).
"""

import torch

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import score_width
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_rows,
    nvfp4_index_score_top16,
)

from .host import SELECT_MAX_ROWS, nvfp4_sparse_decode, warm_up_decode


def route(index, num_reqs: int, query_len: int, num_index_heads: int, num_kv_heads: int,
          compute_type: torch.dtype = torch.bfloat16, split: bool = True) -> str:
    """``"split4"``, ``"split2"``, ``"fused"`` or ``"chain"`` for a decode step (module doc); ``index``: the step's
    ``IndexDecodeTop16`` or None (one GPU)."""
    multi = "one_gpu" if index is None else index.route(num_reqs, query_len, num_index_heads, split)
    if multi != "one_gpu":
        return multi
    if (
        num_index_heads == num_kv_heads
        and num_reqs * query_len <= SELECT_MAX_ROWS.get(num_kv_heads, 0)
        and compute_type in (torch.bfloat16, torch.float16)
    ):
        return "fused"
    return "chain"


def nvfp4_index_sparse_decode(
    index,  # decode_top16.IndexDecodeTop16 (the multi-GPU route's buffers) or None
    iq: torch.Tensor,  # [rows, index heads, 128] e4m3
    index_cache: torch.Tensor,  # NVFP4 index pages [num_blocks, 128, 72] uint8
    index_block_table: torch.Tensor,  # [reqs, max_blocks] int32, the index cache's pages
    index_seq_lens: torch.Tensor,  # [reqs] int32 (the same lengths as ``seq_lens``)
    init_blocks: int,
    local_blocks: int,
    q: torch.Tensor,  # [rows, H, 128] bf16
    kv_cache: torch.Tensor,  # [pages, 2 * Hkv, 128, 72] uint8
    topk_idx: torch.Tensor,  # [Hkv, >= rows, 16] int32, written
    block_table: torch.Tensor,  # [reqs, max_blocks] int32, the KV cache's pages
    seq_lens: torch.Tensor,  # [reqs] int32
    sm_scale: float,
    output: torch.Tensor,  # [rows, H, 128] bf16, written (unless quant_out)
    *,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    decode_query_len: int,
    max_seq_len: int,  # the host's bound of the step's lengths (graphs: max_model_len)
    quant_out: tuple[torch.Tensor, torch.Tensor, int] | None = None,
    split: bool = True,
    compute_type: torch.dtype = torch.bfloat16,
) -> str:
    """Index top-16 + sparse decode of the decode rows (module doc); returns the route taken. ``split`` False keeps
    the step on one GPU (e.g. DBO micro-batches, where no collective orders the multi-GPU exchange); ``compute_type``:
    ``host.nvfp4_sparse_decode``'s. Graph-safe."""
    rows, heads, _ = iq.shape
    num_reqs = seq_lens.shape[0]
    num_kv_heads = kv_cache.shape[1] // 2
    r = route(index, num_reqs, decode_query_len, heads, num_kv_heads, compute_type, split)
    topk = topk_idx[:, :rows]
    extra = {}
    if r in ("split4", "split2"):
        index(iq, index_cache, index_block_table, index_seq_lens, init_blocks, local_blocks, topk,
              query_len=decode_query_len, max_seq_len=max_seq_len, split=split)
    elif r == "fused":
        score = torch.empty((heads, rows, score_width(max_seq_len)), dtype=torch.float32, device=iq.device)
        nvfp4_index_score_rows(iq, index_cache, score, index_block_table, index_seq_lens, init_blocks, local_blocks,
                               query_len=decode_query_len, max_seq_len=max_seq_len)
        extra = dict(index_score=score, index_max_seq_len=max_seq_len)
    else:
        nvfp4_index_score_top16(iq, index_cache, index_block_table, index_seq_lens, init_blocks, local_blocks, topk,
                                query_len=decode_query_len, max_seq_len=max_seq_len)
    nvfp4_sparse_decode(q, kv_cache, topk, block_table, seq_lens, sm_scale, output, k_scale=k_scale,
                        v_scale=v_scale, decode_query_len=decode_query_len, quant_out=quant_out,
                        compute_type=compute_type, **extra)
    return r


def warm_up(
    device: torch.device,
    *,
    num_heads: int,
    num_kv_heads: int,
    query_len: int,
    max_reqs: int,
    quant_min_rows: int | None = None,
    compute_type: torch.dtype = torch.bfloat16,
) -> None:
    """Compile every attention variant the block launches for ``max_reqs`` requests x ``query_len`` rows
    (``host.warm_up_decode`` with the ``index_score`` variants); the index side warms up when its ``IndexDecodeTop16``
    is built."""
    warm_up_decode(device, num_heads=num_heads, num_kv_heads=num_kv_heads, query_len=query_len,
                   max_reqs=max_reqs, quant_min_rows=quant_min_rows, index_select=True, compute_type=compute_type)
