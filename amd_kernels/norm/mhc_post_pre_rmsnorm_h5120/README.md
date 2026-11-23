# mHC seam: post-mix + gates + collapse + RMSNorm (+ MXFP8), hidden 5120

## What it does

One launch per mHC seam (four residual streams, hidden 5120): the post-mix of the previous sub-layer's output, the gate
projection `flatten(R') @ fn^T` (`fn` fp32 `[24, 20480]`), the pre / post / Sinkhorn comb gates, the collapse with the
carried pre gate, and the RMSNorm of the collapse. `reference.py` is the contract (formulas, bf16 rounding points,
tolerance); AITER's HIP `mega_mhc` agrees with it to cosine >= 0.9999997 on every output. The projection multiplies bf16
`R'` by `fn` split into bf16 hi + lo (`pack_gate_weight`, once at load time), fp32 accumulation; reductions have a fixed
order, so repeated calls are bitwise identical. Rows m <= 4096 (tuned at 16 ... 768). No post-mix (`sublayer_out`
absent) and no carried gate (`pre_mix` absent: stream 0) are variants.

Outputs: new residual bf16, `post [m,4,1]`, `comb [m,4,4]`, `next_pre [m,4]` (fp32, the next seam's gates) and the
normed bf16 `layer_input`. `quant=True` adds, in the same launch, MXFP8 codes e4m3 `[m,5120]` and row-major E8M0
bytes `[m,160]` of `layer_input` (`ceil(log2(amax/448))` per 32 columns, the contract of
`norm/rmsnorm_dual_mxfp8_q1280_kv512`). `aux_mean=True` adds the mean of the four new streams (bf16 `[m,5120]`).
Producers and consumers. Attention seam (attn_norm weight): bf16 `layer_input` feeds the bf16 skinny GEMMs
(compressor kv_score, indexer weights_proj); codes + scales feed the fused wq_a + wkv MXFP8 GEMM. FFN seam (ffn_norm
weight): bf16 `layer_input` feeds the router (needs bf16); codes + scales feed the shared-expert gate_up and the
routed MXFP8 experts. `aux_mean` on the seam after layers 37 / 38 yields the draft's auxiliary hidden states with no
standalone post + mean launches (the last one comes from `norm/mhc_post_collapse_rmsnorm_h5120`).

## Entry points

- `mhc_post_pre_rmsnorm`: the seam; calls share one scratch per device and run serially, CUDA-graph safe after one
  call (or the warm-up) outside capture.
- `pack_gate_weight`: `fn` -> the kernel's B operands, once per layer weight.
- `warmup_mhc_post_pre_rmsnorm`: compiles every variant.

Dispatch: `host._config` picks (splits, waves) by row count.

## Performance

One MI355X, shared with other jobs. Graphs of 100 distinct inputs, five replays, rounds A B B A, median us. Baseline:
AITER `mega_mhc` (one HIP launch; `mxfp8` is its fused MXFP8 mode; `+aux` adds `mhc_post` and `mean` launches).

| rows | bf16 HIP | bf16 here | MXFP8 HIP | MXFP8 here | +aux HIP | +aux here |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 8.82 | 8.50 | 8.95 | 8.75 | 15.36 | 9.20 |
| 32 | 9.55 | 8.73 | 9.65 | 8.94 | 16.52 | 9.31 |
| 64 | 9.82 | 9.17 | 10.02 | 9.38 | 17.71 | 9.84 |
| 128 | 10.17 | 9.33 | 10.28 | 9.54 | 19.30 | 10.07 |
| 256 | 14.48 | 11.78 | 14.73 | 12.08 | 25.86 | 12.64 |
| 384 | 15.82 | 16.02 | 16.29 | 16.39 | 29.99 | 19.54 |
| 512 | 19.00 | 21.25 | 19.51 | 22.25 | 35.21 | 24.04 |
| 768 | 28.64 | 29.44 | 29.45 | 31.16 | 50.67 | 32.52 |

Faster up to 256 rows (3-19%), level at 384, slower at 512 and 768 (5-12%); +aux is faster at every size (the baseline
needs two more launches). HBM floor at 8 TB/s: 0.2 us at 16 rows, 9.8 us at 768 (79 MB); graph-node floor about 2.6 us.

## Tests

```bash
python3 -m pytest tests/norm/mhc_post_pre_rmsnorm_h5120 -q
python3 benchmarks/norm/mhc_post_pre_rmsnorm_h5120/benchmark_seam.py
```

Rows 1 ... 4096 (every dispatch bucket and a partial block), the variants against `reference.py` (per-row cosine
>= 0.99999, MXFP8 codes and bytes exact against the bf16 output), repeat bitwise, zero rows, poisoned graph outputs
with changed inputs, warm-up coverage (no compile afterwards), dispatch bounds.
