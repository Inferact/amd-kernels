# MXFP4 DSA indexer entries (gfx950)

## What it does
One shape-agnostic entry per DSA indexer op: it reads the shape from the tensors, calls the shape package that serves
it, and for a call no package serves returns a reason, so the caller runs its fallback. A caller imports no shape
package. Computation, dtypes and numerics are the routed package's (its README and `reference.py`); the entry passes the
arguments through, so its outputs are the package's bit for bit.

| op | key | shape package |
|---|---|---|
| decode scores + per-row top-k | (32, 128, 2048, 64) | `indexer/mxfp4_paged_score_topk_d128_h32_top2048` |
| prefill logits of ragged rows | (32, 128) | `indexer/mxfp4_prefill_score_d128_h32` |
| prefill top-k of each row's window | (2048,) | `indexer/mxfp4_prefill_topk_d128_h32_top2048` |

Keys: decode (heads, head dim) of `q_payload`, top-k of `out_indices`, page rows of `cache`; prefill scores (heads,
head dim) of `q_payload`; prefill top-k the width of `out`.

## Entry points
- `available`: gfx950 with FlyDSL, the load-time check before any shape question.
- `mxfp4_paged_score_topk`, `mxfp4_prefill_score`, `mxfp4_prefill_topk`: the ops, with the package function's
  arguments; a shape off the table raises `ValueError` with the reason.
- `*_unsupported_reason`: the op's arguments -> None or a one-line reason (host checks only, safe inside graph
  capture); the caller asks once per call and falls back on a reason.
- `*_shapes`: the covered keys, for a caller that decides at load, before it has activations.
- `mxfp4_paged_score_topk_kernel`: the routed `amd_kernels.warm_up` entry, for a caller that compiles without
  launching.
- `paged_score_topk_prefer`: whether the fused call beats the unfused scorer + top-k pair (the package's gate).
- `paged_score_topk_preallocate`: the scorer's scratch for the given streams, created before graph capture.
- `paged_score_topk_reset_counts`: zeroes the ready counts a call uses; needed when a batch has padding sequences.
- `prefill_topk_workspace_bytes`: the scratch of one top-k call.

Dispatch: `paged._SHAPE_HOSTS`, `prefill._SCORE_HOSTS`, `prefill._TOPK_HOSTS` (module names, imported on first use);
the scratch helpers route by (top-k, page rows) alone.

## Performance
The entry adds only the dispatch: a graph holds the routed package's kernels. Graph replay of 20 calls on one input,
min of 15 rounds x 5 replays (each graph timed twice, interleaved), us per call, one MI355X.

| call | direct | entry | added |
|---|---|---|---|
| paged score + top-k, 2 x 6 rows, 20480 keys | 13.95 | 13.95 | 0.00 |
| paged score + top-k, 13 x 5 rows, 32768 keys | 30.06 | 30.08 | +0.02 |
| paged score + top-k, 48 x 4 rows, 16384 keys | 32.36 | 32.30 | -0.06 |
| prefill score, 512 rows, 32768 keys | 64.74 | 64.84 | +0.10 |
| prefill score, 2048 rows, 70000 keys | 481.10 | 483.81 | +2.71 (0.6%, run-to-run spread) |
| prefill top-k, 256 rows, 60000 keys | 38.07 | 37.97 | -0.09 |
| prefill top-k, 1024 rows, 131072 keys | 146.32 | 146.39 | +0.07 |

Eager host time added by the entry: 0.9 us (paged), 0.4 us (score), 0.3 us (top-k); the reasons take 5.0 / 4.3 /
2.3 us.

## Tests
```bash
python3 -m pytest tests/indexer/mxfp4_indexer_common -q   # routing (host only), equal to the packages, warm-up + graph
python3 benchmarks/indexer/mxfp4_indexer_common/benchmark_routing.py
```
