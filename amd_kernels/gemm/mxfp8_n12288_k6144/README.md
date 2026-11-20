# MXFP8 GEMM, N 12288 x K 6144

## What it does
The MXFP8 GEMM of weight shape 12288 x 6144, bf16 out. This folder holds the shape's rows, registered with the
dispatcher in [`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._SHAPE_HOSTS`); the kernels, layouts and the
numerics contract are that folder's.

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_shuffled`, `mxfp8_gemm_prequant`,
  `warmup_mxfp8_gemm_weight`).
- `host.ROWS`: this shape's rows (`gemm.parse_rows`); `host.select`: the row for a row count.

## Performance
Every bf16 row measured (benchmarks/gemm/mxfp8_common/sweep_rows.py); the split-K decode rows at 49-224 beat the
rows they replaced at every multiple of 8 and window start, 2-4% slower at 225-256
(benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_common -q` (`test_shape_folders.py`: dispatch through every folder)
