# MXFP8 GEMM, N 25600 x K 6144 (Engram wkv)

## What it does
The MXFP8 GEMM of weight shape 25600 x 6144, bf16 out: the Engram `wkv` projection of DeepSeek-V4.1-Flash at 1-768
rows. This folder holds the shape's rows, registered with the dispatcher in
[`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._SHAPE_HOSTS`); the kernels, layouts and the numerics contract
are that folder's.

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_shuffled`, `mxfp8_gemm_prequant`,
  `warmup_mxfp8_gemm_weight`).
- `host.ROWS`: this shape's rows (`gemm.parse_rows`); `host.select`: the row for a row count.

## Performance
Graph replay, cold weights, µs per call; the rows are AITER's defaults and the measured
bf16 rows (`benchmarks/gemm/mxfp8_common/sweep_rows.py`: decode, staged and AITER configs).

| M | 16 | 128 | 256 | 768 |
|---|---|---|---|---|
| generic fallback row | 38.1 | 66.1 | 116.4 | 143.8 |
| this folder's rows | 27.7 | 41.1 | 54.7 | 142.8 |
| change | -27% | -38% | -53% | -1% |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_common -q` (`test_shape_folders.py`: dispatch through every folder)
