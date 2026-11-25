# MXFP4 top-512 indexer and query RoPE + MXFP4 entries, routed by shape

## What it does

The top-512 keys of each decode row over its MXFP4 paged context (optionally a candidate list and ragged slot lists),
keyed by (index heads, head dim): (32, 128) -> `indexer_mxfp4_h32_d128_top512`; and the indexer query RoPE + MXFP4
quant + weight scaling, keyed by (heads, head dim, rotary dim): (32, 128, 64) -> `q_rope_mxfp4_quant_h32_d128_rot64`.
The top-k blocker also declines the rows where AITER's indexer was faster (above 768). Numerics are the packages'
`reference.py`.

## Entry points

- `mxfp4_indexer_topk`: the dense top-512; `mxfp4_indexer_topk_blocker`, `mxfp4_indexer_shapes`;
  `mxfp4_indexer_candidate_topk` (+ `_blocker`) over a candidate list; `mxfp4_indexer_ragged`,
  `mxfp4_indexer_scores_cols` (shared workspace).
- `prepare_mxfp4_indexer_topk`: the layer's load-time check (heads, page, context bound) and a handle bound to it.
- `q_rope_mxfp4_quant`: the query RoPE + quant; `q_rope_mxfp4_quant_blocker`, `q_rope_mxfp4_quant_shapes`.

Dispatch: `_ROUTES` in `topk.py` and `q_rope.py`, row ranges `_ROWS` in `topk.py`; a shape package is imported on the
first call of its key. Callers import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
