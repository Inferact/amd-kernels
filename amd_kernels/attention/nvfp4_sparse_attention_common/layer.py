# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""NVFP4 sparse attention of one layer (top-k pages per row, NVFP4 K / V pages): the sparse decode, the decode with
its index top-k in the same call, the sparse prefill, and the layer's load-time state (``prepare_nvfp4_sparse_layer``:
the decode indexer and every variant compiled). Routes by (query heads per KV head, head_dim, page, top-k), read from
q [rows, H, head_dim], the cache [pages, 2 * KV heads, page, 72] and the top-k [KV heads, rows, top-k], to the layer
package (``_ROUTES``, imported on the first call of its key)."""

import torch

from amd_kernels import routes
from amd_kernels.routes import Routes, prepare

_NAMES = ("heads_per_kv", "head_dim", "page", "topk")
_HOSTS = {(16, 128, 128, 16): "amd_kernels.attention.nvfp4_sparse_layer_gqa16_d128_p128_top16"}
_DECODE = Routes("nvfp4_sparse_decode", _HOSTS, _NAMES)
_INDEX_DECODE = Routes("nvfp4_index_sparse_decode", _HOSTS, _NAMES)
_PREFILL = Routes("nvfp4_sparse_prefill", _HOSTS, _NAMES)
_LAYER = Routes("nvfp4_sparse_layer", _HOSTS, _NAMES)
_PAGE_BYTES = 72
_I32 = torch.int32


def _key(q, kv_cache, topk):
    if q.dim() != 3 or kv_cache.dim() != 4 or topk.dim() != 3 or kv_cache.shape[1] < 2:
        return None
    return (q.shape[1] // (kv_cache.shape[1] // 2), q.shape[2], kv_cache.shape[2], topk.shape[2])


def _layer_key(num_heads, num_kv_heads, head_dim, block_size, topk_blocks):
    return (num_heads // num_kv_heads if num_kv_heads else 0, head_dim, block_size, topk_blocks)


def nvfp4_sparse_attention_shapes() -> frozenset[tuple[int, int, int, int]]:
    """The (query heads per KV head, head_dim, page, top-k) with a kernel."""
    return _DECODE.shapes()


class Nvfp4SparseLayer:
    """A layer's load-time state from ``prepare_nvfp4_sparse_layer``: ``indexer`` (the decode indexer the index decode
    takes, or None without ``index``) and the decode's MXFP8 output buffers."""

    def __init__(self, key, indexer):
        self.key, self.indexer = key, indexer

    def quant_out_buffers(self, rows: int, n: int, k: int, device):
        """``quant_out`` of a ``rows``-row decode whose output feeds an MXFP8 GEMM with an [n, k] weight, or None
        where the bf16 output is faster."""
        return _LAYER.package(self.key).quant_out_buffers(rows, n, k, device)


def nvfp4_sparse_layer(device, *, num_heads: int, num_kv_heads: int, num_index_heads: int, head_dim: int,
                       block_size: int, topk_blocks: int, query_len: int, max_reqs: int, max_seq_len: int,
                       insert: bool = True, attention: bool = True, index: bool = True, quant_out: bool = True,
                       rank: int | None = None, group=None) -> Nvfp4SparseLayer:
    """The layer's decode indexer (with ``group`` its multi-GPU route, a collective on every rank in order) and every
    insert, decode and prefill variant it launches, compiled (the routed package's ``init``)."""
    key = _layer_key(num_heads, num_kv_heads, head_dim, block_size, topk_blocks)
    indexer = _LAYER.package(key).init(
        device, num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, query_len=query_len,
        max_reqs=max_reqs, max_seq_len=max_seq_len, insert=insert, attention=attention, index=index,
        quant_out=quant_out, rank=rank, group=group)
    return Nvfp4SparseLayer(key, indexer)


def nvfp4_sparse_layer_blocker(device, *, num_heads: int, num_kv_heads: int, num_index_heads: int, head_dim: int,
                               block_size: int, topk_blocks: int, query_len: int, max_reqs: int, max_seq_len: int,
                               insert: bool = True, attention: bool = True, index: bool = True,
                               quant_out: bool = True, rank: int | None = None, group=None) -> str | None:
    """None when ``nvfp4_sparse_layer`` builds this layer's state, else why not."""
    op = _LAYER.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    reason = _LAYER.blocker(_layer_key(num_heads, num_kv_heads, head_dim, block_size, topk_blocks))
    if reason is not None:
        return reason
    if num_heads % max(num_kv_heads, 1) or num_index_heads < 1:
        return f"{op}: {num_heads} query / {num_kv_heads} KV / {num_index_heads} index heads"
    if query_len < 1 or max_reqs < 1 or max_seq_len < 1:
        return f"{op}: query_len {query_len}, max_reqs {max_reqs}, max_seq_len {max_seq_len} (each >= 1)"
    if (rank is None) != (group is None):
        return f"{op}: rank and group go together"
    return None


def prepare_nvfp4_sparse_layer(device, *, num_heads: int, num_kv_heads: int, num_index_heads: int, head_dim: int,
                               block_size: int, topk_blocks: int, query_len: int, max_reqs: int, max_seq_len: int,
                               insert: bool = True, attention: bool = True, index: bool = True,
                               quant_out: bool = True, rank: int | None = None, group=None):
    """``nvfp4_sparse_layer(...)``, or the reason it cannot be built."""
    kw = dict(num_heads=num_heads, num_kv_heads=num_kv_heads, num_index_heads=num_index_heads, head_dim=head_dim,
              block_size=block_size, topk_blocks=topk_blocks, query_len=query_len, max_reqs=max_reqs,
              max_seq_len=max_seq_len, insert=insert, attention=attention, index=index, quant_out=quant_out,
              rank=rank, group=group)
    return prepare(nvfp4_sparse_layer_blocker(device, **kw), lambda: nvfp4_sparse_layer(device, **kw))


def _attention_reason(r, q, kv_cache, topk, tables, out, k_scale, v_scale, q_dtypes=(torch.bfloat16,)):
    op = r.op
    if not routes.available():
        return f"{op}: needs gfx950 with FlyDSL"
    key = _key(q, kv_cache, topk)
    if key is None:
        return (f"{op}: q {tuple(q.shape)} ([rows, H, d]), KV cache {tuple(kv_cache.shape)} ([pages, 2 Hkv, page, "
                f"72]), top-k {tuple(topk.shape)} ([Hkv, rows, k])")
    reason = r.blocker(key)
    if reason is not None:
        return reason
    _, _, page, _ = key
    hkv = kv_cache.shape[1] // 2
    if q.dtype not in q_dtypes or not q.is_contiguous():
        return f"{op}: q {q.dtype} (contiguous {', '.join(str(t) for t in q_dtypes)})"
    if kv_cache.dtype != torch.uint8 or kv_cache.shape[3] != _PAGE_BYTES or kv_cache.stride(3) != 1:
        return f"{op}: KV cache {tuple(kv_cache.shape)} {kv_cache.dtype} (uint8 [pages, {2 * hkv}, {page}, 72])"
    if topk.dtype != _I32 or topk.shape[0] != hkv or topk.shape[1] < q.shape[0]:
        return f"{op}: top-k {tuple(topk.shape)} {topk.dtype} (int32 [{hkv}, >= {q.shape[0]}, k])"
    for name, t in tables:
        if t.dtype != _I32 or t.stride(-1) != 1:
            return f"{op}: {name} {t.dtype} (int32, unit last stride)"
    if out is not None and (out.dtype != torch.bfloat16 or not out.is_contiguous() or out.shape != q.shape):
        return f"{op}: output {tuple(out.shape)} {out.dtype} (contiguous bf16 like q)"
    for t in (k_scale, v_scale):
        if t is not None and t.dtype != torch.float32:
            return f"{op}: k / v scales {t.dtype} (fp32)"
    return None


def nvfp4_sparse_decode(q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, output, k_scale=None, v_scale=None,
                        decode_query_len=1, quant_out=None):
    """Sparse decode attention into ``output`` (or its MXFP8 ``quant_out``) at the package's decode compute dtype."""
    _DECODE.package(_key(q, kv_cache, topk_idx)).sparse_decode(
        q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, output, k_scale=k_scale, v_scale=v_scale,
        decode_query_len=decode_query_len, quant_out=quant_out)


def nvfp4_sparse_decode_blocker(q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, output, k_scale=None,
                                v_scale=None, decode_query_len=1, quant_out=None) -> str | None:
    """None when ``nvfp4_sparse_decode`` serves this call, else why not (host checks only)."""
    if decode_query_len < 1:
        return f"{_DECODE.op}: decode_query_len {decode_query_len}"
    return _attention_reason(_DECODE, q, kv_cache, topk_idx, (("block table", block_table), ("seq lens", seq_lens)),
                             output if quant_out is None else None, k_scale, v_scale)


def nvfp4_index_sparse_decode(index, iq, index_cache, index_block_table, index_seq_lens, init_blocks, local_blocks,
                              q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, output, *, k_scale, v_scale,
                              decode_query_len, max_seq_len, quant_out=None, split=True):
    """The decode rows' index top-k (``index``: ``Nvfp4SparseLayer.indexer``) and their sparse attention in one call;
    returns the route taken."""
    return _INDEX_DECODE.package(_key(q, kv_cache, topk_idx)).index_sparse_decode(
        index, iq, index_cache, index_block_table, index_seq_lens, init_blocks, local_blocks, q, kv_cache, topk_idx,
        block_table, seq_lens, sm_scale, output, k_scale=k_scale, v_scale=v_scale, decode_query_len=decode_query_len,
        max_seq_len=max_seq_len, quant_out=quant_out, split=split)


def nvfp4_index_sparse_decode_blocker(index, iq, index_cache, index_block_table, index_seq_lens, init_blocks,
                                      local_blocks, q, kv_cache, topk_idx, block_table, seq_lens, sm_scale, output, *,
                                      k_scale, v_scale, decode_query_len, max_seq_len, quant_out=None,
                                      split=True) -> str | None:
    """None when ``nvfp4_index_sparse_decode`` serves this call, else why not (host checks only)."""
    op = _INDEX_DECODE.op
    if decode_query_len < 1 or max_seq_len < 1:
        return f"{op}: decode_query_len {decode_query_len}, max_seq_len {max_seq_len}"
    reason = _attention_reason(_INDEX_DECODE, q, kv_cache, topk_idx,
                               (("block table", block_table), ("seq lens", seq_lens),
                                ("index block table", index_block_table), ("index seq lens", index_seq_lens)),
                               output if quant_out is None else None, k_scale, v_scale)
    if reason is not None:
        return reason
    if iq.dim() != 3 or iq.dtype != torch.float8_e4m3fn or iq.shape[0] != q.shape[0]:
        return f"{op}: index q {tuple(iq.shape)} {iq.dtype} (e4m3 [{q.shape[0]}, index heads, d])"
    if index_cache.dtype != torch.uint8 or index_cache.dim() != 3 or index_cache.shape[2] != _PAGE_BYTES:
        return f"{op}: index cache {tuple(index_cache.shape)} {index_cache.dtype} (uint8 [pages, page, 72])"
    return None


def nvfp4_sparse_prefill(q, kv_cache, topk, block_table, query_req_id, query_abs_pos, num_kv_heads, sm_scale, out,
                         k_scale=None, v_scale=None, pack_out=None, quant_out=None):
    """Sparse prefill attention into ``out`` at the package's prefill compute dtype."""
    _PREFILL.package(_key(q, kv_cache, topk)).sparse_prefill(
        q, kv_cache, topk, block_table, query_req_id, query_abs_pos, num_kv_heads, sm_scale, out, k_scale, v_scale,
        pack_out, quant_out)


def nvfp4_sparse_prefill_blocker(q, kv_cache, topk, block_table, query_req_id, query_abs_pos, num_kv_heads, sm_scale,
                                 out, k_scale=None, v_scale=None, pack_out=None, quant_out=None) -> str | None:
    """None when ``nvfp4_sparse_prefill`` serves this call, else why not (host checks only)."""
    if kv_cache.dim() == 4 and num_kv_heads != kv_cache.shape[1] // 2:
        return f"{_PREFILL.op}: num_kv_heads {num_kv_heads}, cache {tuple(kv_cache.shape)}"
    return _attention_reason(_PREFILL, q, kv_cache, topk,
                             (("block table", block_table), ("query req id", query_req_id),
                              ("query abs pos", query_abs_pos)),
                             out, k_scale, v_scale, q_dtypes=(torch.bfloat16, torch.float16))
