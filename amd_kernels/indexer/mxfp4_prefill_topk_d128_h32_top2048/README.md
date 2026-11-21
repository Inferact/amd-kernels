# Prefill indexer top-2048 select (DSA, gfx950)

## What it does

The select half of the DSA prefill indexer, after the FP4 scorer (`indexer/mxfp4_prefill_score_d128_h32`): for each row of an fp32
logits tile `[rows, width]` it returns the window-relative positions of the 2048 largest logits of the row's key window `[ks, ke)`,
ascending, ties to the smaller position, -1 padding rows of fewer than 2048 keys. The result is exact and bit-identical to
`reference.py` (= AITER `top_k_per_row_prefill(stable=True)` with `ks` subtracted, which is what the vLLM call site computes with its
`selected - starts` epilogue); the tile's positions outside the windows are never used. Logits must be finite or +-inf, not NaN; the
row stride is a multiple of 4 floats (the scorer's `pages * 64`), windows up to about 1M keys.

Four launches, one workgroup per row, no host sync: `threshold` (a weighted sample of the window; windows of at most
8192 keys skip it), `stream` (one nontemporal read of the window into an LDS candidate list), `retry` (rows whose count
misses [2048, 8192]) and `select` (exact 2048th largest of each list, ordered compaction; a radix pass over the window for
rows every pass failed on). Design: module docstrings of `kernel.py` and `select.py`.

## Entry points

- `host.topk_rows`: the select for one tile of rows, into a caller-provided int32 `[rows, 2048]`.
- `host.workspace_bytes`: the scratch of a call (cached per device and stream, grown on demand).
- `available`: whether the device is gfx950 and FlyDSL imports.

No dispatch: one compiled set of the four kernels serves every row count and width (`host.py`).

## Performance

Graph replay over logits tiles larger than the LLC, one MI355X, min of 15 (us), one causal request unless noted. The floor is a
read-only kernel streaming the same windows (one workgroup per row, nontemporal 16-byte loads: the ceiling of this layout); TB/s is
window bytes over our time; the stream launch alone runs at 6.1 TB/s at 715 x 500K (234 of the 277 us), the rest is the fixed latency
of the threshold, retry-check and select launches.

| rows x keys | AITER `stable=False` | AITER `stable=True` | ours | speedup | floor | TB/s | % of 6.5 TB/s | % of floor |
|---|---|---|---|---|---|---|---|---|
| 256 x 150K | 62 | 118 | 52 | 1.2x | 26 | 2.9 | 45% | 49% |
| 512 x 150K | 104 | 127 | 84 | 1.2x | 50 | 3.7 | 56% | 59% |
| 715 x 150K | 148 | 181 | 118 | 1.3x | 67 | 3.6 | 56% | 57% |
| 2048 x 150K | 373 | 529 | 316 | 1.2x | 190 | 3.9 | 59% | 60% |
| 512 x 500K | 331 | 359 | 192 | 1.7x | 155 | 5.3 | 82% | 81% |
| 715 x 500K | 477 | 515 | 277 | 1.7x | 208 | 5.2 | 79% | 75% |
| 715 x 790K | 753 | 784 | 407 | 1.9x | 346 | 5.5 | 85% | 85% |
| 715 x 500K, several short requests (mean window 42K) | 68 | 90 | 80 | 0.8x | 30 | 1.5 | 23% | 37% |

Short windows are latency-bound (four launches, ~40 us of threshold / check / select chains): below a mean window of ~60K keys AITER's
single launch is faster.

## Tests

```bash
python3 -m pytest tests/indexer/mxfp4_prefill_topk_d128_h32_top2048 -q
python3 benchmarks/indexer/mxfp4_prefill_topk_d128_h32_top2048/benchmark_topk.py
```
