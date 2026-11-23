# mHC step-end tail: post-mix + collapse + RMSNorm, hidden 5120

## What it does

The tail of the mHC residual after the last decoder layer of a step, one row per workgroup, one launch: the post-mix
of the last sub-layer's output (optional), the collapse of the four streams with the carried pre gate, and the final
RMSNorm. `reference.py` is the contract: fp32 math, `R'` and the collapse rounded to bf16 before they are used, the
norm `bf16(x1 * rsqrt(mean(x1^2) + eps) * w)`. `x1` and `R'` match the reference bit for bit up to the fp32 order of
the post-mix sum; the norm differs by one bf16 ulp in a few elements (per-row cosine >= 0.99999). Reductions have a
fixed order, so repeated calls are bitwise identical. Inputs: residual bf16 `[m, 4, 5120]`, `pre_mix` fp32 `[m, 4]`,
weight bf16 `[5120]`, any m >= 0.

Two options serve the DSpark draft: `keep_residual` also writes the new residual (the hidden-state buffer the draft
reads), `aux_mean` writes the mean of the four new streams (the auxiliary hidden state of the last auxiliary layer).
With `aux_mean` the post-mix, the mean, the collapse and the norm are one launch where the served path runs
`mhc_post`, `mean`, `hc_collapse_triton` and the RMSNorm. Without `sublayer_out` the residual is already the new
residual (the step after an auxiliary layer): collapse + norm only. The auxiliary means of earlier layers come from
`aux_mean` of `norm/mhc_post_pre_rmsnorm_h5120`. Consumer: the target lm_head GEMM reads bf16 `layer_input`.

## Entry points

- `mhc_post_collapse_rmsnorm`: the tail; returns (layer_input, residual or None, mean or None). CUDA-graph safe.
- `warmup_mhc_post_collapse_rmsnorm`: compiles every variant.

One kernel per (post-mix, `keep_residual`, `aux_mean`); the row count is a runtime argument, no dispatch.

## Performance

One MI355X, shared with other jobs. Graphs of 100 distinct inputs, five replays, rounds A B B A, median us. Baselines:
the served chain (`mhc_post` HIP + `hc_collapse_triton` + AITER `rms_norm`; `+aux` adds `mean`) and the fused Triton
`mhc_post_collapse_rms_norm_triton` (post + collapse + norm in one launch, no mean).

| rows | served chain | fused Triton | here | chain + aux | here + aux |
|---:|---:|---:|---:|---:|---:|
| 16 | 6.45 | 4.88 | 3.88 | 10.19 | 3.95 |
| 32 | 6.64 | 4.95 | 3.86 | 10.98 | 4.00 |
| 64 | 7.17 | 5.45 | 4.37 | 12.47 | 4.60 |
| 128 | 8.18 | 6.07 | 4.65 | 14.67 | 4.93 |
| 256 | 10.34 | 7.78 | 5.56 | 18.49 | 6.12 |
| 384 | 13.36 | 12.32 | 8.94 | 21.10 | 9.68 |
| 512 | 16.59 | 14.09 | 9.95 | 25.23 | 10.76 |
| 768 | 23.38 | 19.30 | 14.22 | 33.55 | 15.43 |

HBM floor at 8 TB/s for the 47 MB it moves at 768 rows: 5.9 us, so 41% of it there; below 256 rows it is latency
bound (the launch floor of a graph node here is about 2.6 us).

## Tests

```bash
python3 -m pytest tests/norm/mhc_post_collapse_rmsnorm_h5120 -q
python3 benchmarks/norm/mhc_post_collapse_rmsnorm_h5120/benchmark_final.py
```

Rows 1 ... 1024 with and without the post-mix, the optional outputs against `reference.py`, repeat bitwise, poisoned
graph outputs with changed inputs, warm-up coverage (no compile afterwards).
