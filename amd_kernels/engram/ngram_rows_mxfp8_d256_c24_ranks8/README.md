# N-gram hash + Engram row pull, MXFP8 rows, 24 columns of 256, two layers, eight ranks

## What it does
For the two Engram layers' n-gram tables (one table per layer, 24 hash columns of 256-wide E4M3 rows with E8M0 per-32
scales, sharded in contiguous row ranges over 8 DP ranks), one launch per decode step: vLLM's n-gram hash
(`_hash_ids_kernel`: the batch ids, then the request's 3-token lookback window, through `token_map`; blocked or dead
ids give the pad id; xor of id x multiplier, mod a prime, plus the column offset) and the row lookup, pulling each
row (264 B) straight from its owner rank's table, as the `wkv` GEMM's activation: codes `[M, 6144]` row-major or in
the weight's (16, 16) layout (the row tile of the GEMM's decode row), and E8M0 scales in AITER's shuffled layout,
rows padded to the tile and scales to 32 (127). Replaces vLLM's hash kernel, both layers' BF16 lookups, the
all-gather / all-to-all and the activation quantizer. Inputs: ids, positions (int32 / int64, [M]), `query_start_loc`
int32 [R + 1] (R <= 1024 slots), the runner's `lookback_ids` int32 [R, 3], the tables. Contract (`reference.py`):
bytes are copied, never converted; the hash columns equal vLLM's (`test_vllm_contract.py` runs both), the
codes and scales are bit-exact, repeat runs identical, no atomics. The GEMM consumes the rows as the MXFP8
contract says; vLLM's path re-quantizes BF16 rows, which differs only where a block's scale was not tight.

## Entry points
- `NgramRows.map_tables`: collective, on each of 8 ranks after its shards reach their final allocation; publishes the
  shards, opens the peers' over HIP IPC (no copy); the tables stay alive and unchanged until `close()`.
- `NgramRows(...)`: the same from raw shard addresses (one process; tests, benchmarks).
- `NgramRows.__call__`: hash + pull of a batch into the static buffers, eager or in a CUDA graph, no inter-rank
  synchronization; `tile_rows` = `gemm.prequant_code_tile_rows(M, 25600, 6144)`.
- `NgramRows.activation`: `(codes, scales)` of a layer for `gemm.mxfp8_gemm_prequant`.
- `NgramRows.warm_up`: compiles the one variant; `close`: collective, after every graph is retired.

One kernel; M, tile and dtypes are runtime arguments (`host.py`). The `wkv` GEMM rows (25600 x 6144) are in
`gemm/mxfp8_n25600_k6144`.

## Performance
MI355X, one GPU, graph replay of 100 distinct decode batches (requests of 6 rows, 160 slots), median of 5 rounds x 5
replays, us. The 8 shards are one 12.7 GB allocation per layer at the served size, so every pull is a local-HBM random
read; xGMI is not in these numbers. Baseline: vLLM's Triton `_hash_ids_kernel` + a
`_engram_lookup_kernel` per layer into BF16 rows (its peer-pointer path).

| rows | 16 | 32 | 64 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|
| this entry | 5.8 | 6.3 | 6.4 | 7.0 | 7.4 | 7.8 | 9.0 | 9.4 | 11.7 |
| vLLM Triton chain | 20.3 | 21.4 | 22.7 | 24.1 | 31.4 | 32.4 | 39.2 | 40.0 | 54.7 |

Whole Engram step (pull + wkv GEMM + gated update, two layers) 67.6 us at 16 rows, 196.8 at 384 and 316.1 at 768,
13-43% under vLLM's chain (`benchmark_wkv_chain.py`); the GEMM is 79-86% of it.

## Tests
```bash
python3 -m pytest tests/engram/ngram_rows_mxfp8_d256_c24_ranks8 -q   # one GPU; the 8-rank test skips below 8 GPUs
python3 -m pytest tests/engram/ngram_rows_mxfp8_d256_c24_ranks8/test_ngram_rows_ranks8.py -q   # needs 8 GPUs
python3 benchmarks/engram/ngram_rows_mxfp8_d256_c24_ranks8/benchmark_ngram_rows.py
python3 benchmarks/engram/ngram_rows_mxfp8_d256_c24_ranks8/benchmark_wkv_chain.py
```
