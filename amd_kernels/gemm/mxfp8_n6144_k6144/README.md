# MXFP8 GEMM, N 6144 x K 6144

## What it does
The MXFP8 GEMM of weight shape 6144 x 6144, bf16 out: the dense gate_up at I 3072 per rank and the dense down
projection at I 6144 per rank (the same GEMM, so one set of rows). This folder holds the shape's rows, registered with
the dispatcher in [`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._SHAPE_HOSTS`); the kernels, layouts and the
numerics contract are that folder's.

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_shuffled`, `mxfp8_gemm_prequant`,
  `warmup_mxfp8_gemm_weight`).
- `host.ROWS`: this shape's rows (`gemm.parse_rows`).
- `host.select`: the row for a row count (`gemm.select_from_rows`).

## Performance
Split-K decode at 33-256 rows beats the decode rows at every row count; K-wave 2 decode rows at 257-320, 385-448 and
641-1024, `4wave_nt` 256 x 128 at 1025-1280 (`benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py`,
`sweep_rows.py`).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_common/test_shape_folders.py -q` (dispatch through every folder)
- `python3 -m pytest tests/gemm/mxfp8_common/test_decode_splitk.py -q` (the split-K rows of this shape)
