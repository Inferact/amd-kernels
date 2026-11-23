# Gemma RMS norm + RoPE + e4m3 KV insert entry, routed by shape

## What it does

One layer's attention input with Gemma-normed q / k heads (weight `1 + w`): per-head Gemma RMS norm, partial NeoX
RoPE and the e4m3 K / V cache insert. The key (head_dim, rotary_dim) is read from `q_norm_weight` [head_dim] and
`cos_sin_cache` [rows, rotary_dim]; (128, 64) -> `norm_rope_fp8_insert_d128_rot64`. Head counts and page size are the
package's compile modes. The plain-norm contract lives in `norm_rope_insert_common`; this family is kept apart because
the norm differs. Arguments, dtypes and numerics are the package's (its `reference.py`); the entry writes the same bits
as the direct call. `*_blocker` reads only shapes, dtypes, strides and host values (safe inside graph capture).

## Entry points

- `gemma_norm_rope_insert_common.available`: gfx950 with FlyDSL importable.
- `gemma_norm_rope_insert_common.gemma_norm_rope_insert_rope_blocker`: load-time check of a layer's RoPE (rotary
  dims, NeoX, cos / sin cache).
- `gemma_norm_rope_insert_common.gemma_qkv_norm_rope_insert`: q into the caller's buffer, K / V of the tokens with
  slots into the cache; `gemma_qkv_norm_rope_insert_blocker`, `gemma_qkv_norm_rope_insert_shapes`,
  `gemma_qkv_norm_rope_insert_kernel` (the routed warm-up entry).

Dispatch: `_ROUTES` in `qkv.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/attention/gemma_norm_rope_insert_common/test_routing.py -q   # host only
python3 -m pytest tests/attention/gemma_norm_rope_insert_common -q
```
