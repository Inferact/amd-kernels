# Routed experts, MXFP8 x MXFP4, H 5120 I 2304 E 16 top 3 (gfx950)

## What it does

The routed experts of one DSpark draft layer under EP8: 16 local experts of 128, 3 per token, hidden 5120, intermediate
2304, run on the token rows the rank receives (5 draft rows per request, 1-1024 tokens, each with its 3 local expert
slots). The shape variant of `moe/mxfp4_h5120_i2304_e48_top6` (the target layers' 48 experts, top 6): the same
kernels with this shape's constants and dispatch tiers. Input: MXFP8 activations (e4m3 codes `[n, 5120]`, row-major
E8M0 `[n, 160]`), int32 `topk_ids` `[n, 3]` in `[0, 16)`, fp32 `topk_weights`; the weights are MXFP4 codes with E8M0
per 32 K (`w13 [16, 4608, 2560]` gate rows then up rows, `w2 [16, 5120, 1152]`, checkpoint layouts, preshuffled once
by `ExpertWeights.from_row_major`). Output: bf16 `[n, 5120]`.

Contract (`reference.py`): per route, `gate, up = x @ w13[e].T` in fp32, clamp-SwiGLU (limit 10), the activation quantized
to MXFP8 per 32 columns, `slot = bf16(w_route * (q @ w2[e].T))`, `out = bf16(sum of the 3 slots in slot order, fp32)`;
cosine >= 0.99999 against it, two runs bit for bit, no atomics.

Producer and consumer: the activations are what the layer's FFN norm + MXFP8 quantizer writes (codes and row-major
scales, vLLM's `QuantizedActivation`); `topk_ids` / `topk_weights` are
[`moe/router_sqrtsoftplus_top3_h5120_e128`](../router_sqrtsoftplus_top3_h5120_e128/README.md)'s outputs after the
dispatch maps a rank's expert ids to `[0, 16)`. The intermediate (clamp-SwiGLU + quantization) is written by GEMM1's
epilogue in GEMM2's operand layout, so no quantizer sits between the GEMMs; the output is bf16 for the shared-expert
add and the next seam.

## Entry points

- `ExpertWeights.from_row_major`: the checkpoint tensors in the kernel layouts; once, before graph capture.
- `mxfp4_experts`: the routed experts of a token block (`return_slots=True`: the bf16 slots `[n, 3, 5120]` instead of
  their sum). An `amd_kernels.warm_up` entry: the first call of a mode compiles every dispatch tier. The one-launch
  GEMM1 + GEMM2 + combine shares one ready-counter arena per device: two calls must not run concurrently on a device.

`host.DISPATCH` picks the sort block, tiles and ring depths by token count (swept per tier at 16-1024 tokens).

## Performance

One MI355X, graph replay of 40 distinct inputs cycling over 4 weight sets (0.3 GB each, as three draft layers' weights
stream from HBM), rounds A B B A, median us per call. Baseline: AITER's FlyDSL a8w4 `fused_moe` (sort + two GEMMs with
the MXFP8 requant fused; the fastest of its kernels per count, `sweep_baseline.py`; its atomic stage 2
sums in arrival order). This is the compute of one rank: the served path's dispatch and combine are not in either
column. The tiles are the target layers' kernels (`moe/mxfp4_h5120_i2304_e48_top6`), tuned here by dispatch only.

| tokens | 16 | 40 | 80 | 160 | 320 | 640 | 1024 |
|---|---|---|---|---|---|---|---|
| AITER fused_moe, uniform | 64.0 | 68.4 | 73.0 | 86.6 | 116.1 | 139.9 | 174.6 |
| this entry, uniform | 51.9 | 54.3 | 62.2 | 75.2 | 86.0 | 112.2 | 158.5 |
| change | -19% | -21% | -15% | -13% | -26% | -20% | -9% |
| AITER fused_moe, skewed | 57.6 | 69.2 | 74.6 | 91.7 | 117.3 | 141.9 | 177.7 |
| this entry, skewed | 49.0 | 56.1 | 64.6 | 76.8 | 91.8 | 117.6 | 156.8 |
| change | -15% | -19% | -13% | -16% | -22% | -17% | -12% |

## Tests

```bash
python3 -m pytest tests/moe/mxfp4_h5120_i2304_e16_top3 -q
python3 benchmarks/moe/mxfp4_h5120_i2304_e16_top3/benchmark_experts.py [--tokens 160 640] [--layers 4]
python3 benchmarks/moe/mxfp4_h5120_i2304_e16_top3/sweep_baseline.py  # AITER's fastest kernels per count
```

One GPU: tokens 1-1024 (uniform and skewed routing, an empty and a hot expert) against the reference for the sum and
the slots, repeat runs, graph replay with new inputs, no compile after the warm-up.
