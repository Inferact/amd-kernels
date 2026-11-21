# MoE router, sqrt-softplus top-3 of 128 experts, H 5120 (gfx950)

## What it does

The router of one DSpark draft layer for decode rows: the gate GEMM and the biased top-3 in two launches. The 128-expert
top-3 shape of the router of [`moe/router_sqrtsoftplus_top6_h5120_e384`](../router_sqrtsoftplus_top6_h5120_e384/README.md)
(same kernels, tiles tuned for this shape). Input: x bf16 `[M, 5120]` (unit column stride, any row stride), gate weight
bf16 `[128, 5120]` row-major, bias fp32 `[128]`, fp32 routed scaling factor (1.5), optional bool `is_padding [M]`.
Outputs: `topk_weights` fp32 `[M, 3]`, `topk_ids` int32 `[M, 3]` in descending `score + bias` order.

Contract (`reference.py`): `logits = x @ W^T` (bf16 products, fp32 accumulate), `score = sqrt(logit if logit > 20 else
log(1 + exp(logit)))`, the 3 experts with the largest `score + bias` (ties: the lowest expert id), weights
`((score + bias) - bias) * (scale / sum)` of the picks, summed in slot order; padding rows get id -1 and weight 0. Given
the same logits, the ids and weights equal vLLM's `topk_hash_softplus_sqrt` bit for bit (tests). The logits sum k-slices
of the GEMM in a fixed order that depends on the row-count bucket, so a row's ids differ from the served pair's only where
the 3rd and 4th `score + bias` are within 1e-5. Graph-capture safe, no atomics, bitwise repeatable.

Producer and consumer: x is the draft layer's FFN-norm output (bf16 `[M, 5120]`, the same row the routed experts'
MXFP8 quantizer reads); `topk_ids` (int32) and `topk_weights` (fp32) `[M, 3]` are the routing inputs of
[`moe/mxfp4_h5120_i2304_e16_top3`](../mxfp4_h5120_i2304_e16_top3/README.md) once the dispatch has mapped the expert ids
of the rank's 16 experts to `[0, 16)`; nothing converts them in between.

## Entry points

- `router_topk`: ids and weights of a row block; outputs and the fp32 partials scratch may be the caller's buffers.
- `warmup_router_topk`: compiles every row-count bucket, with and without padding and the shuffled gate, before capture.
- `shuffle_gate`: the gate in the load-time layout `[expert tile, k step, lane, 8]` (a bit-exact permutation, once at
  weight load) for `router_topk` with `weight_shuffled`; same contract, faster from 160 rows, ties or loses at 80-96.

`host._tile` picks the GEMM tile by row count (`host._TILES`, `host._TILES_SHUF`); the second launch sums the k-slice
partials and picks the top 3 with one wave per token.

## Performance

One MI355X, graph replay of 100 distinct inputs, median of 5 x 5 replays, rounds A B B A, us per call, mean of two
runs; rows = 5 per request. Baseline: vLLM's `fp32_router_gemm_dispatch` + `topk_hash_softplus_sqrt`. Both launches
start with a cold L2: latency-bound below 320 rows (the 1.3 MB weight is 0.2 us of HBM time).

| rows | 16 | 40 | 80 | 160 | 320 | 640 | 768 |
|---|---|---|---|---|---|---|---|
| served GEMM + top-k | 9.14 | 15.59 | 13.98 | 14.05 | 15.82 | 17.88 | 17.95 |
| this entry (plain gate) | 4.94 | 5.34 | 5.79 | 7.04 | 10.56 | 14.13 | 14.13 |
| this entry, `weight_shuffled` | 4.74 | 5.17 | 5.76 | 6.09 | 8.43 | 11.99 | 12.16 |

Limit: the two launches' latency (2 graph nodes x 1.59 us = 3.2 us) to 160 rows, the cold read of the call's bytes
(5.2 us) at 768; achieved 4.73 / 6.08 / 12.15 us at 16 / 160 / 768 rows (68 / 52 / 43 %), the GEMM operand-traffic
bound above 320 rows.

## Tests

```bash
python3 -m pytest tests/moe/router_sqrtsoftplus_top3_h5120_e128 -q
python3 benchmarks/moe/router_sqrtsoftplus_top3_h5120_e128/benchmark_router.py
python3 benchmarks/moe/router_sqrtsoftplus_top3_h5120_e128/sweep_tiles.py --rows 16 192  # re-tunes host._TILES
```

One GPU: 21 row counts (1-2049) against the reference and the served op pair, ties, above-switch logits, padding, row
strides, non-finite inputs, caller buffers, graph replay with poisoned outputs, determinism, no compile after warm-up.
