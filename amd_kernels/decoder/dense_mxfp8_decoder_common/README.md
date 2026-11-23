# Dense MXFP8 decoder-stack entries on a TP group, routed by shape

## What it does

The dense decoder layers around the caller's attention on a TP group with MXFP8 linears: the split input fc, the
embedding + norm + quant, the all-reduce + norm tails, the MLP blocks and the context K / V chain, held by one block
built collectively. The block's key (hidden, world) is read from the fc weight [hidden, 5 * hidden] and the group:
(6144, 4) -> `dense_mxfp8_h6144_i3072_ranks4`, (6144, 2) -> `dense_mxfp8_h6144_i6144_ranks2`; a layer's weights route
by (hidden, I) of its down weight. The ops are the block's methods (`fc`, `forward`, `store_context_kv`, `capture`).
Numerics are the package's `reference.py`.

## Entry points

- `dense_mxfp8_decoder_common.available`: gfx950 with FlyDSL importable.
- `dense_mxfp8_decoder_common.dense_decoder_layer`: one layer's weights for `forward`.
- `dense_mxfp8_decoder_common.dense_decoder_stack_blocker`: load-time check of the whole stack.
- `dense_mxfp8_decoder_common.prepare_dense_decoder`: the block or the reason (also `dense_decoder`,
  `dense_decoder_blocker`, `dense_decoder_shapes`).
- `dense_mxfp8_decoder_common.dense_decoder_context_kv_blocker`: whether `store_context_kv` serves the bound caches.

Dispatch: `_ROUTES`, `_LAYERS` in `stack.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/decoder/dense_mxfp8_decoder_common -q   # host only
```
