# MXFP4 indexer top-512 over decode rows (32 heads x 128 dims, 64 / 128-entry pages)

## What it does

For each decode row (`next_n` rows of a request, each with the context length it may see) the logits of AITER's
`paged_mxfp4_mqa_logits`, logit(t) = sum_h w_h relu(q_h . k_t) (e2m1 codes, e8m0 scales per 32 dims, exact products,
fp32 sums), and its 512 best keys, without writing the `[rows, context]` logits: positions ascending, -1 padded, ties
to the lower position, bit-identical run to run (`reference.py`: the exact fp64 selection; the kernels' sums can swap
two keys within a few ulps at the 512th rank). One walk reads each key once for all rows of a request and keeps the
maximum logit of every 8-token block; a row's 512 best keys lie in its 512 best blocks, so a select keeps a bucketed
superset of those blocks, scores their tokens again with the same arithmetic and selects exactly.
`indexer_candidate_topk` selects from a row's 2048 candidate blocks (the two-level indexer's consumer layers); the
candidate source also writes that list (newest key's block + the 2047 best by block maximum). Both can emit the sparse
attention's ragged slot lists. Four bit-identical paths (the tests compare them; each kernel's docstring holds its
design): token (up to 16384 entries, `kernel_tokens`), block (up to 262144, `kernel_select`), long (rows above 262144,
`kernel_blocks`), split candidate consumer (`kernel_cand_logits`).

Up to `MAX_ENTRIES` = 1048576 entries per row (`blocker` refuses more). `max_ctx` is the static bound of the launch and
must be at least the longest row (a longer row is truncated). A bound above 262144 adds about 5-10 us of long-path
launches even when no row is long, and a long row is a chain of five dependent kernels (~57 us): fall back to AITER for
the dense and source layers above 262144 entries below about 64 rows a step; the consumer layers win at every size.
Producers: `indexer/q_rope_mxfp4_quant_h32_d128_rot64` (`q`, `q_scale`, `weights`) and the index pages of
`attention/compressor_insert_d512_idx128_rot64`; `attention/sparse_mla_decode_fp8kv_h64_d512_rot64` reads
`Ragged.slots` / `indptr` as its `extra_indices` / `extra_indptr`.

## Entry points

- `indexer_topk`: dense layer: the top-512 positions of every row; optionally the candidate list and the ragged lists.
- `indexer_candidate_topk`: the top-512 of a row among its candidate blocks, optionally the ragged lists.
- `indexer_block_scores`: the 8-token block maxima alone (a test and benchmark hook).
- `Ragged`: the ragged outputs and the attention cache's page table; `blocker`: why a layer is out of range.
- `random_inputs`, `*_reference`, `topk_matches`: the torch contract and its test inputs.

All are `amd_kernels.warm_up` entries; CUDA-graph safe once the block-maxima workspace exists (call once eagerly per row
count). Dispatch: `host.py` (path and tier by `max_ctx`, kernels by `next_n` and page size, the grid by rows).

## Performance

Dense layer (`indexer_topk` + ragged lists), one MI355X, graph replay of 100 distinct decode steps (6 rows a request,
request i of N at 0.75 + 0.25 (i + 0.5) / N of the context), A B B A, median us per call, `baseline / this (speedup)`;
baseline: vLLM's path over AITER (schedule / 4, `paged_mxfp4_mqa_logits`, `top_k_per_row_decode`, ragged lists).

| context | 18 rows | 66 | 192 | 384 | 768 |
|---|---|---|---|---|---|
| 4096 | 21.8 / 13.5 (1.62x) | 23.0 / 14.5 (1.59x) | 24.0 / 17.4 (1.38x) | 32.2 / 21.8 (1.47x) | 46.8 / 30.6 (1.53x) |
| 8192 | 22.8 / 16.9 (1.35x) | 25.5 / 19.9 (1.28x) | 28.9 / 22.8 (1.27x) | 43.8 / 33.0 (1.33x) | 68.3 / 48.7 (1.40x) |
| 16384 | 25.8 / 25.0 (1.03x) | 31.2 / 28.5 (1.10x) | 38.8 / 35.2 (1.10x) | 58.7 / 54.7 (1.07x) | 98.5 / 89.8 (1.10x) |
| 32768 | 30.8 / 26.2 (1.17x) | 38.1 / 30.1 (1.27x) | 52.6 / 44.3 (1.19x) | 88.2 / 81.7 (1.08x) | 149.4 / 146.4 (1.02x) |
| 65536 | 36.4 / 27.9 (1.30x) | 54.9 / 39.4 (1.40x) | 80.0 / 66.2 (1.21x) | 142.8 / 127.7 (1.12x) | 257.4 / 237.0 (1.09x) |
| 131072 | 43.0 / 34.6 (1.24x) | 71.8 / 54.9 (1.31x) | 140.8 / 113.3 (1.24x) | 257.6 / 217.3 (1.19x) | 488.8 / 421.0 (1.16x) |

Limit: package power (the walk runs at 1.30 kW, 2.10 GHz) and VALU / MFMA overlapping at most ~30% of the MFMA time;
at 192 rows / 130k the walk reaches 2.7 TB/s, 38% of the 7.14 TB/s HBM stream and 80% of its compute-alone time.

## Tests

```bash
python3 -m pytest tests/indexer/indexer_mxfp4_h32_d128_top512 -q
python3 benchmarks/indexer/indexer_mxfp4_h32_d128_top512/benchmark_indexer.py --mode dense|source|candidates
```
