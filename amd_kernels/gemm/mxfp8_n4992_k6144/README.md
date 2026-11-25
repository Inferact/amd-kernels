# MXFP8 GEMM, N 4992 x K 6144 (QKV projection, TP2)

## What it does
The QKV projection of the MiniMax-M3.1-Flash sparse attention layer at TP2: N 4992 = q (32 heads) | k (2) | v (2) |
index-q (2) | index-k (1, replicated) x 128, K 6144, bf16 out. Every bf16 row of the shape and its persistent kernel,
registered with the dispatcher in [`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._ROUTES`); layouts and
the numerics contract are that folder's. Callers: the fused norm + quant producer writes (16, 16) codes in the tile
`prequant_code_tile_rows` names (decode rows) or row-major codes (persistent rows); the deferred partials feed the
split-K reduce + norm / RoPE / NVFP4 insert of `attention/norm_rope_nvfp4_insert_d128_rot64_p128` (32 / 2 / 2 heads).

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_prequant`, `prequant_code_tile_rows` +
  `mxfp8_gemm_shuffled`, which returns `SkPartials` on a deferred row); `warmup_mxfp8_gemm_weight` compiles every row.
- `host.select`: this shape's config for a row count (decode and split-K decode rows to 768, then the persistent
  tile of the fewest rounds).
- `host.select_deferred`: the split-K partial row that the fused norm / RoPE / insert chain reduces.
- `host.persistent_tile` / `host.persistent_plan`: the persistent kernel's tile for a row count, and its launches
  (one, or a row split into whole rounds of one tile plus the rest on another).
- `host.decode_launcher`: the decode kernel in the grouped XCD order for the rows of three or more m-blocks.
- `host.LAUNCHERS`: the launchers of the persistent kernel (one launch, row split).

The `host.py` selectors are the dispatch; `mxfp8_common`'s `gemm.py` calls them.

## Performance
One MI355X. Graph replay, layer 39 TP2 rank 0 weights in 14 cold copies (> 400 MB), up to 100 seeded inputs,
3 x 5 alternated rounds, median µs. GEMM: pre-quantized codes in (the producer's layout) -> bf16; chain: + the
TP2 norm / RoPE / NVFP4 insert (`mxfp8_gemm_qkv_norm_rope_nvfp4_insert`). Baseline: vLLM's TP2 fallback, Triton
`tl.dot_scaled` (`_M31_TILES[(4992, 6144)]`) on row-major codes (+ the insert). Roofline: 6.5 TB/s, 5 PF.

| M | Row | GEMM µs | Baseline µs | Chain µs | Baseline µs | % of GEMM roofline |
|---|---|---|---|---|---|---|
| 8 | decode 16x32 (chain: split-2 partials) | 7.58 | 10.58 | 10.27 | 13.38 | 64% |
| 32 | decode 32x32 (chain: split-2 partials) | 8.83 | 11.67 | 11.07 | 14.69 | 56% |
| 64 | split-K decode 64x64, K split 3 | 9.56 | 14.25 | 12.90 | 17.86 | 53% |
| 96 | split-K decode 96x64, K split 3 | 10.57 | 19.71 | 14.15 | 23.05 | 48% |
| 128 | split-K decode 2 x 64x128, K split 3 | 11.30 | 21.12 | 14.91 | 24.59 | 46% |
| 176 | split-K decode 2 x 96x128, K split 3 | 13.12 | 23.04 | 17.77 | 26.91 | 40% |
| 256 | split-K decode 2 x 128x128, K split 3 | 15.71 | 36.37 | 20.89 | 40.69 | 35% |
| 384 | decode 192x48 | 20.99 | 35.36 | 26.56 | 40.24 | 28% |
| 512 | decode 128x96, grouped | 24.54 | 64.21 | 30.57 | 70.21 | 26% |
| 768 | decode 256x64, grouped | 30.31 | 65.81 | 36.71 | 72.33 | 31% |
| 1024 | persistent 192x128 | 39.61 | 94.58 | 47.25 | 102.01 | 32% |
| 2048 | persistent 256x192 | 56.50 | 114.10 | 67.30 | 124.54 | 44% |
| 4096 | persistent 256x192 | 101.48 | 184.16 | 120.16 | 201.22 | 50% |
| 8192 | persistent 256x192 + 256x128 split | 192.12 | 379.86 | 230.31 | 408.90 | 52% |
| 16384 | persistent 256x192 | 372.23 | 815.05 | 442.13 | 869.98 | 54% |
| 32768 | persistent 256x192 | 758.59 | 2068.21 | 888.06 | 2175.64 | 53% |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n4992_k6144 -q`
- `python3 benchmarks/gemm/mxfp8_n4992_k6144/benchmark_n4992_k6144.py --checkpoint <model>`
- `python3 benchmarks/gemm/mxfp8_n4992_k6144/sweep_n4992_k6144.py --decode 16,256 --chain 64 --prefill 4096`
