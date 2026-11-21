# MXFP4 paged index scorer + per-row top-2048, head dim 128, 32 heads (gfx950)

## What it does
The DSA decode indexer's `score_paged` + `_select(..., score_rows)` (vLLM `dsa.py`) in one launch; the fp32 logits
`[rows, width]` are never written. Queries fp4 `[rows, 32, 64]` + e8m0 scales, index pages `[pages, 64, 68]` uint8,
head weights fp32, a block table, causal `row_ends` (1 - 6 rows a sequence, <= 256 rows, <= 131072 keys a rank). Out: per
row the 2048 best keys, unordered (ties at the 2048th score in any order), `-1` / `-inf` past the live keys; rows past
`query_start_loc[-1]` are padding. Scores are AITER `flydsl_pa_mqa_logits_fp4_rowgroup`'s bit for bit; the selection is
exact. Contract: `reference.py` (torch dequant, logits, `check_selection`, scores within 1e-4 relative).
Chunk workgroups (`lists.py`) score a chunk into LDS and keep each row's top list; merge workgroups (`merge.py`, one a
row) wait on their sequence's ready count and select the top 2048 from the lists. Protocol: module docstring of `fused.py`.

## Entry points
- `mxfp4_paged_score_topk`: the fused scorer + top-2048, an `amd_kernels.warm_up` entry, CUDA-graph safe; `uniform=True`
  when every sequence has `max_query_len` rows; scratch ~145 MB a device and stream.
- `prefer_fused`: the measured gate (stock is 5 - 10% faster only for >= 24 sequences of <= 8192 keys).
- `cache_views`: payload and scale views of a merged index cache, without a copy.

`host.plan` picks the rows a workgroup scores and the chunks a sequence takes by batch size.

## Performance
Graph replay, cold pages (24 inputs over a 3 GiB pool), min of 15, us per call, one GPU; stock = AITER row-group scorer into
fp32 logits + `top_k_per_row_decode`. Ceiling = max(bytes / 6.4 TB/s, flops / 8.24 PF) + one launch (1.57) + flag (0.8) + 2 cold
loads (0.5) = 3 - 11 us. The op is latency-bound (rocprofv3, whole kernel: VALU busy 3.5% / 20% / 33%, MFMA busy 0.5% / 4% /
6%, wave occupancy 6% / 16% / 24%, waves waiting 72% / 62% / 54% of their cycles at 2x6x15K / 20x4x32K / 48x4x32K).

| Sequences x rows | Keys a rank | Stock | Fused | Speedup | % of ceiling |
|---|---|---|---|---|---|
| 2 x 6 | 15360 / 20480 / 94208 | 22.1 / 21.2 / 43.4 | 15.1 / 14.7 / 28.6 | 1.46x / 1.44x / 1.51x | 21 / 22 / 16 |
| 3 x 4 | 8192 / 32768 / 94208 | 15.3 / 25.2 / 44.0 | 14.4 / 17.6 / 28.2 | 1.06x / 1.43x / 1.56x | 22 / 22 / 20 |
| 8 x 5, 10 x 4 | 16384, 32768 | 23.7, 30.9 | 20.3, 25.7 | 1.17x, 1.20x | 21, 23 |
| 13 x 5 | 32768 / 65536 | 36.5 / 55.1 | 30.2 / 45.5 | 1.21x / 1.21x | 23 / 24 |
| 20 x 4 / 8 x 4 | 32768 / 94208 | 37.7 / 52.7 | 33.7 / 36.5 | 1.12x / 1.44x | 27 / 27 |
| 48 x 4 | 8192 / 16384 / 32768 | 22.5 / 35.7 / 57.1 | 25.0 / 34.0 / 54.1 | 0.90x / 1.05x / 1.05x | 28 / 32 / 33 |

A decode step's dependent chain (one launch, dependent loads, MFMA, two selections, a flag, the merge) is ~8 us at 2 x 6
rows, and the chunk workgroups of 20 - 48 sequences are VALU- and latency-bound at 20 - 33% busy.

## Tests
```bash
python3 -m pytest tests/indexer/mxfp4_paged_score_topk_d128_h32_top2048 -q  # shapes, ladder, hot / tied, streams, warm-up
python3 benchmarks/indexer/mxfp4_paged_score_topk_d128_h32_top2048/benchmark_score_topk.py [--quick]
```
