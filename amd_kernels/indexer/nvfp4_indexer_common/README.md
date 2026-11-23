# NVFP4 lightning-indexer prefill top-k entry, routed by shape

## What it does

The prefill rows' top-k index blocks of a lightning indexer over NVFP4 index pages: scorer + top-k of every prefill
row (ragged requests after their cached prefix) in one call. The key (index_head_dim, page, top-k) is read from iq
[rows, heads, d] e4m3, the index cache [pages, page, 72] and `topk_idx` [heads, rows, k]; (128, 128, 16) ->
`indexer_nvfp4_d128_p128_top16` (1, 2 or 4 index heads per GPU). Numerics are the package's `reference.py`; the
entry writes the direct call's ids.

## Entry points

- `nvfp4_indexer_common.available`: gfx950 with FlyDSL importable.
- `nvfp4_indexer_common.index_prefill_top16_layer_blocker`: load-time check of a model's indexer (heads, head dim,
  block size, top-k).
- `nvfp4_indexer_common.index_prefill_top16`: the op; `index_prefill_top16_blocker`, `index_prefill_top16_shapes`,
  `warm_up_index_prefill_top16` (compile before serving).

Dispatch: `_ROUTES`, `_LAYER` in `prefill.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/indexer/nvfp4_indexer_common -q   # host only
```
