# Fused expert-parallel MoE, H5120 I2304 top6, 8 ranks (gfx950)

## What it does

One launch per rank for a whole EP MoE layer of DeepSeek-V4.1-Flash (384 experts over 8 ranks, `epr` = 48 local experts
by default; any `epr` <= 63 is a shape parameter, 48, 49, 56 and 63 are tested): MXFP8 quantize of bf16 rows (or MXFP8
rows as given), dispatch to the owner ranks (one row per distinct destination), route records, GEMM1 with
`silu(min(g, 10)) * clamp(u, -10, 10)` and MXFP8 requant in the epilogue, GEMM2 with route weights and a return of each
weighted row to the token's rank as its tile finishes, deterministic combine. Out: bf16 `[rows, 5120]`. Weights MXFP4 in
AITER's shuffle layouts. No output atomics or fences, bitwise repeatable, graph capturable, counters reset by the kernel.
Per-rank row counts may differ (0 = idle rank .. 1024). `topk_ids` are physical rank-major ids (owner = `id // epr`,
-1 pads contribute zero), so an EPLB placement only renumbers ids. Contract: `reference.py` over all ranks (cosine >=
0.99999). Assumptions: one `epr` for all ranks, one node (xGMI, IPC pools), one call in flight per device, every rank
calls every layer (idle ranks with 0 rows); a call returns only after every rank's GEMM2 finished.

## Entry points

- `ExpertWeights.from_row_major`: one layer's local experts in the kernel layouts, once, before graph capture.
- `ExpertParallel`: one rank's IPC pool, control words and scratch (a collective, created before capture, one per
  process for every layer); calling it runs the layer on the rank's bf16 or MXFP8 rows.
- `ExpertParallel.warm_up`: compiles every token-count tier (a collective, before capture).

`host.DISPATCH` / `host.sized_tokens` pick the tile tier by token count: tiles are sized for twice the rank's tokens
(a rank owning hot experts receives more), 32-row blocks from 9 tokens, 8-wave 64-row blocks from 49.

## Performance

8 x MI355X, graph replay of 100 distinct inputs, us per call (max over ranks), median of 4 A B B A samples in one
session (spread under 2 % except T 512-768 and single outliers); zipf = Zipf 1.2 expert popularity, hot = every route
on rank 0. Baselines: staged = AITER MegaMoE dispatch + experts + combine (as in CI); #7788 = AITER MegaMoEV2's fused
routes (ROCm/aiter#7788, `mega_moe_small_m.py`).

| T per rank | 1 | 16 | 32 | 64 | 128 | 192 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|
| fused uniform | 46.9 | 145.0 | 155.2 | 168.2 | 178.8 | 187.8 | 228.3 | 306.0 | 354.8 |
| staged / #7788 | 88 / 47.3 | 316 / 162 | 209 / 166 | 229 / 169 | 232 / 182 | 345 / 221 | 385 / 266 | 453 / 356 | 566 / 560 |
| fused zipf | 41.7 | 104.4 | 135.1 | 169.1 | 227.9 | 279.6 | 425.8 | 570.2 | 768.7 |
| #7788 zipf | 44.2 | 114.4 | 150.2 | 204.5 | 283.8 | 334.7 | 492.5 | 624.5 | 932.5 |
| fused hot | 109.9 | 154.6 | 206.7 | 239.5 | 410.2 | 570.9 | 1033.6 | 1246.5 | 1813.0 |
| #7788 hot | 112.7 | 167.8 | 227.4 | 311.8 | 513.5 | 621.5 | 1131 | 1495 | 2200 |

No loss at any T and routing (uniform T 1 / 64 / 128 tie, 1.01-1.02x). Unequal rows per rank (one full rank + 1/8
elsewhere, ramp, 192 / 384 tokens): 1.10-1.22x.

Limit (uniform, phases from `AKR_EP_TRACE=1`): to T 192 the HBM weight stream binds, 87-94 % of the measured
7.15 TB/s inside the GEMM loops but 71-77 % for the whole call (arrival, tail and the first tile leave HBM idle); from
T 384 the return stores bind (T 768: 47 MB in 122 us = the 360 GB/s xGMI ceiling; whole call 44-62 % of HBM).
Overlapping dispatch / return with the weight stream (expert-group dispatch, weight prefetch, GEMM2 mixed into GEMM1)
did not pay off.

## Tests

```bash
python3 -m pytest tests/moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 -q
torchrun --nproc-per-node 8 benchmarks/moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8/benchmark_experts.py
AKR_EP_TRACE=1 python3 benchmarks/moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8/benchmark_timeline.py --world 8
```

Needs 8 GPUs (one process per GPU through `tests/multi_gpu8.py`; skipped with fewer, `AKR_OVERSUBSCRIBE=1` runs a
plumbing check only); the test takes 37 s warm. `benchmark_timeline.py` prints the per-rank phase timeline.
