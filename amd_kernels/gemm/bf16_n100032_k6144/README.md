# bf16 GEMM, N 100032 x K 6144 (the TP2 lm_head vocabulary shard)

## What it does
`out [M, 100032] = x [M, 6144] @ W^T`, bf16 operands, fp32 sums in ascending K, one bf16 rounding: the TP2 vocabulary
shard of the MiniMax-M3.1-Flash lm_head (`lm_head.weight` bf16 [200064, 6144], rank r holds rows
[100032 r, 100032 (r + 1))). One tensor serves the target lm_head (verify rows) and the DSpark draft lm_head
(requests x 7 rows). W layout: `row` (the checkpoint's shard) or `shuf` (a bit-exact permutation once at load, so one
DMA reads 1 KB in one piece); both give the same bits. No split-K, partials or output atomics; every tile variant
gives the same bits (one wave's K-ascending MFMA chain per output). Contract (`reference.py`): the fp64 product
rounded once to bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3, deterministic; NaN / inf propagate as in an fp32
sum. Caller: `sampling/batch_sharded_markov_v200064_h6144_r256_ranks2` (target and draft logits).

## Entry points
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout; replace the parameter once
  before graph capture, keep the target / draft alias, free the row-major copy.
- `bf16_gemm_n100032_k6144`: the logits of any row count (launches of 2048 rows), graph-safe, no host sync; `out` may
  be a column slice of a wider row-major tensor (a vocabulary half of [rows, 200064]). With a device row map: the
  mapped rows in map order (`rows`), or (with an exchange slot of `collectives/all_to_all_bf16_n100032_ranks2`) the
  mapped rows written into the send slot and published, the caller then runs the exchange's pull. An
  `amd_kernels.warm_up` entry: the first call of each mode (layout x contiguous / rows / publish) compiles its 15 tile
  variants.
- `warmup_bf16_gemm_n100032_k6144`: the same warm-up for one mode, explicitly (publish: a private slot).
- `supports`: whether a weight is the [100032, 6144] bf16 contiguous shard (either layout) and there is no bias.

`host.tile_config` picks the kernel, tiles, waves and ring by row count.

## Performance
One MI355X. Graph replay, `shuf`, 50 distinct inputs (the 1.23 GB weight exceeds the MALL), ten replays, five
alternated rounds, median µs. vLLM = `rocm_unquantized_gemm_impl` on the row-major shard (wvSplitK at 1 row,
hipBLASLt / AITER above; VLLM_ROCM_USE_AITER=1). Roofline = max(bytes / 6.5 TB/s, flops / 2.5 PF). `publish` = the
mapped-row kernels with system-scope stores into the exchange slot and the ticket (the `rows` form stores plainly).

| M | vLLM | contiguous | publish | roofline | contiguous / roofline |
|---:|---:|---:|---:|---:|---:|
| 1 | 182.8 | 176.9 | 178.4 | 189.1 | 1.07 |
| 8 | 199.4 | 177.4 | 179.1 | 189.4 | 1.07 |
| 16 | 207.6 | 177.9 | 180.0 | 189.6 | 1.07 |
| 64 | 254.4 | 184.1 | 188.8 | 191.2 | 1.04 |
| 96 | 268.7 | 195.0 | 198.1 | 192.2 | 0.99 |
| 128 | 278.6 | 205.7 | 211.2 | 193.3 | 0.94 |
| 144 | 292.0 | 241.0 | 243.0 | 193.8 | 0.80 |
| 160 | 297.4 | 255.0 | 262.1 | 194.3 | 0.76 |
| 192 | 304.9 | 280.3 | 289.0 | 195.4 | 0.70 |
| 224 | 315.5 | 308.9 | 310.3 | 196.4 | 0.64 |
| 256 | 325.6 | 332.3 | 335.1 | 197.5 | 0.59 |
| 320 | 345.0 | 448.2 | 459.5 | 199.6 | 0.45 |
| 384 | 391.3 | 505.7 | 516.7 | 201.7 | 0.40 |
| 512 | 570.4 | 592.7 | 596.3 | 251.7 | 0.42 |
| 1024 | 951.2 | 1035.0 | 1044.2 | 503.5 | 0.49 |
| 2048 | 1927.9 | 2042.1 | 2058.4 | 1007.0 | 0.49 |

It beats vLLM through 224 rows (the owned rows of up to 56 requests in batch-sharded sampling); from 256 rows
hipBLASLt / AITER are faster.

## Tests
- `python3 -m pytest tests/gemm/bf16_n100032_k6144 -q` (`test_publish.py`: 2 GPUs)
- `python3 benchmarks/gemm/bf16_n100032_k6144/benchmark_gemm.py [M ...]`
