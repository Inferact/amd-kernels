# bf16 GEMM entries, routed by shape

## What it does

`x @ W^T` in bf16 (fp32 accumulate) for a wide weight (lm_head, a vocabulary shard), optionally in the package's load-
time shuffled layout, keyed by the weight's (N, K): 50016 x 6144, 100032 x 6144, 129280 x 5120. And one GEMM that
writes its first columns in fp32 and the rest in bf16, keyed by (K, bf16 columns): (5120, 32). `bf16_gemm_blocker`
also declines the rows of 129280 x 5120 where vLLM's lm_head GEMM was faster (it wins at 113-128 and 496-512 rows).
Numerics are the packages' `reference.py`.

## Entry points

- `bf16_gemm`: the GEMM; `bf16_gemm_blocker`, `bf16_gemm_weight_blocker` (load time), `bf16_gemm_shapes`,
  `bf16_gemm_shuffle_weight[_]`, `warmup_bf16_gemm`, `prepare_bf16_gemm` (shuffled in place, compiled).
- `bf16_gemm_f32_bf16`: the fp32 + bf16 split GEMM; `bf16_gemm_f32_bf16_blocker`, `bf16_gemm_f32_bf16_shapes`,
  `bf16_gemm_f32_bf16_shuffle_weight`.

Dispatch: `_ROUTES`, `_SPLIT` and the row ranges `_ROWS` in `host.py`; a shape package is imported on the first call
of its key. Callers import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/gemm/bf16_gemm_common -q   # host only
```
