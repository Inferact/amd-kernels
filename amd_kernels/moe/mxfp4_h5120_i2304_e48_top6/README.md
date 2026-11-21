# MXFP8 x MXFP4 routed experts, H5120 I2304 E48 top6 (gfx950)

## What it does

One expert-parallel rank's routed experts of a 384-expert top-6 MoE (48 local experts, hidden 5120, intermediate 2304;
the count is the weights' first dimension, up to 63: 48, 52 and 56 are tested), FlyDSL only. Weights MXFP4 (e2m1 +
e8m0 per 32 K) in the layouts of AITER's `shuffle_weight_a16w4` / `shuffle_scale_a16w4` (a MegaMoE rank's tensors serve
as they are); activations MXFP8 (e4m3 `[n, 5120]` + e8m0 `[n, 160]`, row-major, as an upstream norm / dispatch
quantizer writes them) or bf16 (quantized in the routing launch).
Per route: GEMM1 on the scaled FP4 x FP8 MFMA (fp32), `silu(min(g, 10)) * clamp(u, -10, 10)` in
fp32, requantized to MXFP8 (e8m0 = exponent of `ceil_pow2(amax * fp32(1/448))`) inside the GEMM1 epilogue in the
operand layout GEMM2 loads (no quant or dequant launch between the GEMMs), GEMM2, route weight, one bf16 rounding. A
token's six slots are summed in slot order (fp32, one rounding). No output atomics: every element has one writer, runs
are bitwise repeatable, graph replay equals eager. Contract: `reference.py` (cosine >= 0.99999; 4 % of slots 1 ulp off).

Interfaces. In: the router's `topk_ids` / `topk_weights` as local ids (an id outside [0, E), such as the router's -1
or another rank's expert, contributes zero), or rows already sorted by expert with `expert_offsets` (a dispatch's
receive buffer). Out: bf16 `[n, 5120]`, or the route-weighted rows in input order for the combine that returns them.

## Entry points

- `ExpertWeights.from_row_major`: the checkpoint tensors in the kernel layouts, once, before graph capture.
- `mxfp4_experts`: the experts of `n <= 1024` tokens (MXFP8 or bf16 rows, top-6 ids and weights): bf16 `[n, 5120]`, or
  with `return_slots` the six weighted slots. A `warm_up` entry per mode (bf16 or MXFP8, slots or combined).
- `mxfp4_experts_rows`: rows sorted by expert (`n_rows <= 6144`, MXFP8 or bf16, row weights, E + 1 offsets): the weighted
  output row of each input row; rows from `expert_offsets[E]` on are not written. A `warm_up` entry per input kind.
- `kernel.compile_experts` / `host.DISPATCH`: GEMM1, GEMM2 and the combine in one launch after the sort / block-table
  launch; the token count -> tile table (block rows 16-128, prefetch, ring depth, 4 or 8 waves). One call at a time per
  device (per-device ready counters, `host.ready_counters`).

## Performance

One MI355X (shared GPU, A B B A rounds). Graph replay of 100 distinct activations and routes (6 distinct experts per
token; skewed: odds ~ 1/(rank + 1), hottest expert 6.5x the mean), weights beyond the LLC, median us per call, MXFP8
input. Baseline: AITER's FlyDSL a8w4 `fused_moe` (sort, stage 1 with fused requant, atomic stage 2), its fastest stage
kernels per count (`benchmark_baseline_sweep.py`); outputs agree to cosine 0.999994.

| T tokens (6T rows) | 16 | 32 | 64 | 128 | 192 | 384 | 768 |
|---|---|---|---|---|---|---|---|
| this entry, uniform | 123.8 | 135.8 | 144.2 | 149.0 | 154.8 | 172.1 | 221.0 |
| baseline, uniform | 136.6 | 153.2 | 160.7 | 170.9 | 165.5 | 186.1 | 271.3 |
| change | -9.4% | -11.4% | -10.3% | -12.8% | -6.5% | -7.5% | -18.5% |
| this entry, skewed | 101.0 | 126.4 | 142.2 | 161.2 | 178.6 | 205.1 | 265.7 |
| baseline, skewed | 111.0 | 137.4 | 160.1 | 183.2 | 183.2 | 220.6 | 297.1 |
| change | -9.0% | -8.0% | -11.2% | -12.0% | -2.5% | -7.0% | -10.6% |

Limit (uniform): the HBM weight stream, measured ceiling 7.15 TB/s (6.6-7.0 for this weight layout in FlyDSL);
achieved 6.0-6.6 TB/s (84-92 %) at T 16-192, 5.6 TB/s (79 %) at T 384 (MFMA pipe 58 % busy, vector-memory issue stalls
20 %), 4.7 TB/s (65 %) and 1.9 PF of padded work (42 % of the 4.65 PF fp8 MFMA rate) at T 768.

## Tests

```bash
python3 -m pytest tests/moe/mxfp4_h5120_i2304_e48_top6 -q
python3 benchmarks/moe/mxfp4_h5120_i2304_e48_top6/benchmark_experts.py --dist skew  # or uniform; --no-baseline
```
