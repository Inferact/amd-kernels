# Engram entries, routed by shape

## What it does

The n-gram hash + MXFP8 row pull of the Engram layers over the owner ranks of one node (an object built once, called
per step), keyed by (ranks, layers, hashed columns): (8, 2, 24) -> `ngram_rows_mxfp8_d256_c24_ranks8`; and the gated
update of the residual streams, keyed by (streams, hidden): (4, 5120) -> `gated_post_bf16_h5120_hc4`. Numerics are the
packages' `reference.py`.

## Entry points

- `ngram_rows`: the row-pull object; `ngram_rows_blocker`, `ngram_rows_shapes`, `prepare_ngram_rows` (the object or
  the reason).
- `gated_post`: the gated update; `gated_post_blocker`, `gated_post_shapes`.

Dispatch: `_NGRAM_ROWS`, `_GATED_POST` in `host.py`; a shape package is imported on the first call of its key. Callers
import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
