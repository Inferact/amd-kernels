# NVFP4 sparse attention entries (decode, index + decode, prefill, layer state), routed by shape

## What it does

Sparse attention of one layer over NVFP4 K / V pages, each row attending its top-k pages: the decode, the decode with
the index top-k in the same call, and the prefill, at the package's served compute dtypes; and the layer's load-time
state (`Nvfp4SparseLayer`: the decode indexer, every insert / decode / prefill variant compiled, the decode's MXFP8
output buffers for o_proj). The key (query heads per KV head, head_dim, page, top-k) is read from q [rows, H, d], the
cache [pages, 2 * KV heads, page, 72] and the top-k [KV heads, rows, k] (the layer state takes it as arguments);
(16, 128, 128, 16) -> `nvfp4_sparse_layer_gqa16_d128_p128_top16`. Numerics are the routed packages' `reference.py`.

## Entry points

- `nvfp4_sparse_attention_common.available`: gfx950 with FlyDSL importable.
- `nvfp4_sparse_attention_common.prepare_nvfp4_sparse_layer`: the layer state or the reason (also
  `nvfp4_sparse_layer`, `nvfp4_sparse_layer_blocker`); `Nvfp4SparseLayer.indexer`, `.quant_out_buffers`.
- `nvfp4_sparse_attention_common.nvfp4_sparse_decode`, `nvfp4_index_sparse_decode`, `nvfp4_sparse_prefill`: the ops,
  each with its `*_blocker`; `nvfp4_sparse_attention_shapes` (covered keys).

Dispatch: `_DECODE`, `_INDEX_DECODE`, `_PREFILL`, `_LAYER` in `layer.py`; a shape package is imported on the first
call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/attention/nvfp4_sparse_attention_common -q   # host only
```
