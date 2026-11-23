# Compressor insert entry, routed by shape

## What it does

The compressor pool + RMSNorm + RoPE + latent insert (e4m3) and, through `wk`, the indexer key store (MXFP4) of the
group-closing tokens, in one launch. The key (latent dim, index key dim, rotary dim) is read from the main cache, `wk`
and the cos / sin cache; (512, 128, 64) -> `compressor_insert_d512_idx128_rot64`. Numerics are the package's
`reference.py`.

## Entry points

- `compressor_insert`: the compressor tail and key store; `compressor_insert_blocker`, `compressor_insert_shapes`.
- `compressor_shuffle_wk`: `wk` in the layout the op reads (load time).

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
