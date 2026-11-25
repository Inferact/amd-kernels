# RoPE + fp8 KV insert entry, routed by shape

## What it does

The query RoPE + e4m3 quant and the KV RoPE + e4m3 insert into a paged cache in one launch. The key (heads, head dim,
rotary dim) is read from `q` and the cos / sin cache; (64, 512, 64) -> `rope_fp8_kv_insert_h64_d512_rot64`. Numerics
are the package's `reference.py`.

## Entry points

- `rope_fp8_kv_insert`: the query and KV RoPE + e4m3 insert; `rope_fp8_kv_insert_blocker`,
  `rope_fp8_kv_insert_shapes`.

Dispatch: `_ROUTES` in `host.py`; a shape package is imported on the first call of its key. Callers import the names
from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
