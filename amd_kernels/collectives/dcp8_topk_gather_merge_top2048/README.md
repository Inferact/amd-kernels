# DCP8 top-2048 score gather + merge + owned-slot CSR, one launch

## What it does

Decode-time sparse attention under DCP8 keeps the KV cache sharded over eight ranks. Each rank has its local top-2048
candidates of every row (`scores` fp32 `[rows, 2048]`, padding -inf; `local_idx` int32 `[rows, 2048]`, -1 = none). The
op gathers the eight score planes, takes the global top-2048 of each row and emits this rank's share as physical KV rows
in CSR form (`slots` int32 packed over rows, `indptr` int32 `[rows + 1]`), in place of the all-gather of the scores
plus `merge_dcp_scores` (AITER top-k values, a Triton owned-slot select, a Triton compaction).

`reference.py` is the contract: integer-exact against vLLM's `merge_dcp_scores` (the global top-2048 by score, ties at
the 2048th score in rank-major position order, NaN below -inf, only live candidates emitted, local index `j` mapped to
`block_table[row, j // page_size] * block_stride + j % page_size`). Every other element of `slots` is unspecified.

One workgroup per row and one all-to-all round in the common case: each rank pushes its keys in the highest bins of a
histogram (a fixed 2 KB block per peer), every rank finds the exact threshold from the received keys, and rows whose
planes are too skewed take a second round (and, if needed, a third), chosen identically on all ranks. No flags: receivers poll the
pushed words. Protocol and hazards: module docstring of `kernel.py`.

## Entry points

- `TopkGatherMerge`: collective constructor over eight ranks of a gloo group (maps the IPC pool); one per stream.
- `TopkGatherMerge.warmup`: collective before capture; compiles the single variant (rows are a runtime argument).
- `TopkGatherMerge.__call__`: the op; the caller passes the outputs, so capture allocates nothing.
- `TopkGatherMerge.close`: collective, after every graph using it is retired.

All ranks call in the same order with the same row count (at most 256).

## Performance

Eight MI355X, CUDA-graph replay of 100-call graphs over distinct inputs (larger than LLC), MAX over ranks, min of 15
rounds, us. Stock = AITER IPC all-gather of the scores + `merge_dcp_scores`, same harness. Floor = max of the latency
floor 6.7 us (1.57 launch + 2 xGMI hops + selection, measured ceilings) and bytes pushed / 343 GB/s.

| rows | stock | this entry | speedup | pushed MB | floor | % of floor |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 27.5 | 13.5 | 2.04 | 0.17 | 6.7 | 50 |
| 24 | 30.3 | 15.0 | 2.01 | 0.35 | 6.7 | 45 |
| 40 | 33.2 | 16.2 | 2.05 | 0.58 | 6.7 | 41 |
| 65 | 38.6 | 17.9 | 2.16 | 0.94 | 6.7 | 37 |
| 80 | 41.2 | 18.7 | 2.20 | 1.15 | 6.7 | 36 |
| 128 | 50.1 | 20.6 | 2.43 | 1.84 | 6.7 | 32 |
| 192 | 61.1 | 24.3 | 2.51 | 2.76 | 8.1 | 33 |

Not within 10% of the floor: one hop (1.8 us), two LDS radix passes (2.4 us), the cut and push (3 us) and the row-count
chain (1.5 us) are serial.

## Tests

```bash
python3 -m pytest tests/collectives/dcp8_topk_gather_merge_top2048 -q     # 1 GPU (reference vs vLLM) + 8 GPUs
# served row counts 1-256, random ties, all-equal rows, graph replay, bit-identical reruns
python3 benchmarks/collectives/dcp8_topk_gather_merge_top2048/benchmark_gather_merge.py   # 8 GPUs, needs vLLM
```
