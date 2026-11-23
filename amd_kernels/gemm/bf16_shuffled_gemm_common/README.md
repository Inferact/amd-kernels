# bf16 GEMM entry over a wide weight in its load-time shuffled layout, routed by shape

## What it does

`x [M, K] @ W^T` in bf16 with fp32 accumulate for a wide weight (a vocabulary shard) that the caller permutes once at
load into the package's layout (bit-exact, in place). The key (N, K) is the weight's shape: (50016, 6144) ->
`bf16_n50016_k6144`, (100032, 6144) -> `bf16_n100032_k6144`, (129280, 5120) -> `bf16_n129280_k5120`. Any row count.
Numerics are the package's `reference.py`; the entry returns the direct call's bits.

## Entry points

- `bf16_shuffled_gemm_common.available`: gfx950 with FlyDSL importable.
- `bf16_shuffled_gemm_common.prepare_bf16_shuffled_gemm_weight`: the weight shuffled in place, or the reason;
  `bf16_shuffled_gemm_weight_blocker` (load-time check), `bf16_shuffled_gemm_shuffle_weight_`.
- `bf16_shuffled_gemm_common.bf16_shuffled_gemm`: the op; `bf16_shuffled_gemm_blocker`, `bf16_shuffled_gemm_shapes`,
  `warm_up_bf16_shuffled_gemm`.

Dispatch: `_ROUTES` in `gemm.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/gemm/bf16_shuffled_gemm_common/test_routing.py -q   # host only
python3 -m pytest tests/gemm/bf16_shuffled_gemm_common -q
```
