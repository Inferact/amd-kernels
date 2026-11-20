# MXFP8 GEMM, N 6144 x K 4096 (o_proj, TP2)

## What it does
The attention output projection of MiniMax-M3.1-Flash at TP2: K 4096 = 32 heads x 128, N 6144, bf16 out; the target's
attention layers and the DSpark draft's attention (64 heads / 2 ranks). TP2 has no sequence-parallel `o_proj_full`, so
this shape serves every prefill chunk too (to 32768 rows). The rows and kernels of this shape, registered with the
dispatcher in [`../mxfp8_common`](../mxfp8_common/README.md) (`gemm._SHAPE_HOSTS`); layouts and the numerics contract
are that folder's. A producer of o_proj codes writes them in the tile `prequant_code_tile_rows` names (decode rows to
768, row-major above).

## Entry points
- The common entries (`mxfp8_linear`, `mxfp8_gemm`, `mxfp8_gemm_shuffled`, `mxfp8_gemm_prequant`,
  `warmup_mxfp8_gemm_weight`).
- `host.select`: this shape's config for a row count, for every row count (bf16 out; split-K decode rows at
  49-224); it is the dispatch.
- `host.select_prequant`: the decode row for producer codes.
- `host.decode_launcher`: the owner-read decode kernel of the 96-column two-K-wave rows (None: the common one).
- `host.LAUNCHERS`: the launcher of each kernel `select` names (the 128x192 and one-tile 256x192 copies at K 4096,
  the K-generic persistent 256x192 of `gemm/mxfp8_n6144_k8192`).

## Performance
One MI355X. Graph replay, layer 39 o_proj TP2 rank 0 weights (`[:, :4096]`) in 16 cold copies (> 400 MB), seeded
inputs, 15 alternated rounds, median us per GEMM. Baseline: vLLM's Triton `tl.dot_scaled` linear, the TP2 fallback
(`_mxfp8_dot_scaled_gemm` on codes, `_mxfp8_dot_scaled_linear` on bf16). Codes: the producer's MXFP8 codes
(`mxfp8_gemm_prequant`); bf16: `mxfp8_linear` (quantizer included). Roofline: max(bytes / 6.5 TB/s, flops / 5 PF).

| M | codes us | baseline us | bf16 us | baseline us | roofline us |
|---|---|---|---|---|---|
| 1 / 8 | 5.84 / 5.90 | 8.92 / 9.02 | 5.77 / 6.33 | 11.12 / 11.23 | 4.00 |
| 32 / 64 | 6.77 / 8.09 | 14.66 / 22.30 | 8.99 / 10.56 | 17.02 / 24.80 | 4.07 / 4.16 |
| 128 / 256 | 9.54 / 13.11 | 18.14 / 31.90 | 12.19 / 16.63 | 19.04 / 33.23 | 4.32 / 4.64 |
| 320 / 384 | 14.44 / 15.62 | 39.43 / 44.88 | 18.03 / 19.36 | 43.24 / 47.61 | 4.81 / 4.97 |
| 512 / 768 | 18.38 / 25.75 | 46.54 / 65.12 | 22.73 / 30.48 | 46.26 / 68.32 | 5.29 / 7.73 |
| 1024 / 2048 | | | 32.24 / 47.24 | 67.80 / 84.81 | 10.31 / 20.62 |
| 4096 / 8192 | | | 91.34 / 160.39 | 167.74 / 348.64 | 41.23 / 82.46 |
| 16384 / 32768 | | | 316.21 / 621.37 | 735.65 / 1811.77 | 164.9 / 329.9 |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n6144_k4096 -q`
- `python3 benchmarks/gemm/mxfp8_n6144_k4096/benchmark_n6144_k4096.py --checkpoint <model> --form codes`
- `python3 benchmarks/gemm/mxfp8_n6144_k4096/benchmark_n6144_k4096.py --checkpoint <model> --form bf16`
