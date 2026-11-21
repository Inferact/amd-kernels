# FP4 prefill indexer scorer (32 heads, head dim 128, MXFP4), gfx950

## What it does

MQA logits of ragged prefill rows over the staged index key pages of a chunk, the first half of the DSA prefill indexer (the
top-2048 select follows, AITER `top_k_per_row_prefill`): `logit[r, n] = sum_h weight_scale * w[r, h] * relu(q[r, h] . k[n])` for
`ks[r] <= n < ke[r]`. Queries are MXFP4 (E2M1 values, E8M0 scale per 32, preshuffled as `quantize_queries` writes them), keys the
`stage_keys` payload / scale pages (identity page table), weights bf16; fp32 accumulate, fp32 logits [rows, width]. Eight
consecutive rows share one workgroup, so a key page is read once for eight rows; whole pages are stored (positions of the window's
pages outside a row's window are unspecified). The contract is `reference.py`: rel. 1e-5 of the row max against the dequantized
fp32 math, 2 ulp against AITER's `flydsl_pa_mqa_logits_fp4_prefill`, bit-reproducible.

## Entry points

- `host.score_rows`: the scorer for one tile of rows; replaces `compute_prefill_schedule` + `flydsl_pa_mqa_logits_fp4_prefill`
  (no Triton schedule kernels, no per-width compile). Grid sizing is `host.grid_for`.
- `available`: whether the device is gfx950 and FlyDSL imports.

## Performance

Graph replay over key sets larger than the LLC, one idle MI355X, us (see the module docstring of `kernel.py` for the cycle model;
the ceilings are measured: 8.2 PF register-resident FP4 MFMA, no VALU/MFMA overlap on a SIMD).

| rows x keys | AITER scorer | `score_rows` | speedup | TFLOP/s | % of 8.2 PF MFMA peak | % of cycle model |
|---|---|---|---|---|---|---|
| 512 x 150K | 320 | 268 | 1.19x | 2348 | 29% | 96% |
| 512 x 500K | 1087 | 894 | 1.22x | 2346 | 29% | 96% |
| 715 x 150K | 473 | 388 | 1.22x | 2270 | 28% | 93% |
| 715 x 500K | 1756 | 1299 | 1.35x | 2257 | 28% | 92% |
| 1280 x 150K | 833 | 679 | 1.23x | 2328 | 28% | 95% |
| 2048 x 150K | 1298 | 1050 | 1.24x | 2413 | 29% | 99% |

The 8.2 PF peak is not reachable here: ReLU + weighted head sum is VALU work that does not overlap the MFMAs (two full-rate FMAs
per score), which sets the model's 2.4-2.8 PF limit; the kernel runs at 92-99% of it.

## Tests

```bash
python3 -m pytest tests/indexer/mxfp4_prefill_score_d128_h32 -q
python3 benchmarks/indexer/mxfp4_prefill_score_d128_h32/benchmark_score.py
```
