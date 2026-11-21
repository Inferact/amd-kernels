# MoE router, sqrt-softplus top-6 of 384 experts, H 5120 (gfx950)

## What it does

The router of one MoE layer for decode rows: the gate GEMM and the biased top-6 in two launches. Input: x bf16
`[M, 5120]` (unit column stride, any row stride), gate weight bf16 `[384, 5120]` row-major, bias fp32 `[384]`, fp32
routed scaling factor (1.5), optional bool `is_padding [M]`. M >= 0 (tuned for decode rows 16-768; larger M works).
Outputs: `topk_weights` fp32 `[M, 6]`, `topk_ids` int32 `[M, 6]` in descending `score + bias` order.

Contract (`reference.py`): `logits = x @ W^T` (bf16 products, fp32 accumulate), `score = sqrt(logit if logit > 20 else
log(1 + exp(logit)))`, the 6 experts with the largest `score + bias` (ties: the lowest expert id), weights
`((score + bias) - bias) * (scale / sum)` of the picks, summed in slot order (vLLM's kernel decodes the weight from the
argmax key, hence the round trip); padding rows get id -1 and weight 0 (vLLM's `VLLM_MOE_SKIP_PADDING`). This is vLLM's
`topk_hash_softplus_sqrt` with `renormalize`, without hash tables or the image-token `bias_vl` (a language-model-only
checkpoint has neither). Given the same logits, the ids and weights equal that op bit for bit (tests: normal and
above-switch logits; exp is the fast exp2, log and sqrt the accurate ones). The logits sum k-slices of the GEMM in a
fixed order that depends on the row-count bucket, so their last fp32 bits (about 3e-7 relative to the largest logit)
differ from the served GEMM's: a row's ids differ from the served pair's only where the 6th and 7th `score + bias`
are within 1e-5. Graph-capture safe, no atomics, bitwise repeatable.

## Entry points

- `router_topk`: ids and weights of a row block; outputs and the fp32 partials scratch may be the caller's buffers.
- `warmup_router_topk`: compiles every row-count bucket, with and without padding and the shuffled gate, before capture.
- `shuffle_gate`: the gate in the load-time layout `[expert tile, k step, lane, 8]` (a bit-exact permutation, once at
  weight load; every weight fragment of the GEMM is one coalesced 1 KB run) for `router_topk` with `weight_shuffled`;
  same contract, its own tile table. The plain layout stays supported.

`host._tile` picks the GEMM tile by row count (`host._TILES`, `host._TILES_SHUF`: MFMA tiles per workgroup, k-slices,
waves, XCD grid); the second launch sums the k-slice partials and picks the top 6 with one wave per token.

## Performance

One MI355X, graph replay of 100 distinct inputs, median of 5 x 5 replays, rounds A B B A, us per call, mean of two
runs (spread <= 0.1 us). Baseline: vLLM's `fp32_router_gemm_dispatch` (its ROCm fp32 router kernel at the smallest M,
hipBLASLt above) + `topk_hash_softplus_sqrt`. Every launch starts with a cold L2.

| rows | 16 | 32 | 48 | 64 | 96 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| served GEMM + top-k | 14.89 | 19.25 | 17.21 | 17.34 | 17.23 | 17.25 | 19.39 | 19.98 | 24.84 | 24.52 | 28.12 |
| this entry (plain gate) | 6.37 | 7.01 | 8.34 | 7.96 | 9.29 | 10.14 | 10.73 | 14.00 | 16.47 | 19.60 | 22.45 |
| this entry, `weight_shuffled` | 6.09 | 6.67 | 6.86 | 7.27 | 8.32 | 8.57 | 9.43 | 10.70 | 14.73 | 15.27 | 17.52 |

Limit: two launches' latency (2 graph nodes x 1.59 us) + the cold read of W and x (3.1 / 4.7 us at 192 / 768 rows);
achieved 9.43 / 17.5 us (55 / 51 %): the GEMM is operand-traffic bound through the CU L1s, the top-6 launch is 3.6 us.

## Tests

```bash
python3 -m pytest tests/moe/router_sqrtsoftplus_top6_h5120_e384 -q
python3 benchmarks/moe/router_sqrtsoftplus_top6_h5120_e384/benchmark_router.py
python3 benchmarks/moe/router_sqrtsoftplus_top6_h5120_e384/sweep_tiles.py --rows 16 192  # re-tunes host._TILES
```

One GPU: 20 row counts (1-2049) against the reference and the served op pair, ties, above-switch logits, padding, row
strides, non-finite inputs, caller buffers, graph replay with poisoned outputs, determinism, no compile after warm-up.
