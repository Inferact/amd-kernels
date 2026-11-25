# Sparse MLA decode entry, routed by shape

## What it does

Attention of every decode row over its window and top-k key lists of an e4m3 latent cache, then the inverse RoPE and
the MXFP8 quant of the output (or bf16). The key (query heads, head dim) is read from `q`; (64, 512) ->
`sparse_mla_decode_fp8kv_h64_d512_rot64`. Numerics are the package's `reference.py`. `sparse_mla_decode_blocker` takes
the op's arguments plus `keys_per_row` (the tensors do not show it) and also declines the rows where AITER's sparse
MLA + quant was faster (above 768).

## Entry points

- `sparse_mla_decode`: the decode attention with its MXFP8 epilogue; `sparse_mla_decode_blocker`,
  `sparse_mla_decode_shapes`.

Dispatch: `_ROUTES` and the row ranges `_ROWS` in `host.py`; a shape package is imported on the first call of its key.
Callers import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
