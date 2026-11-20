# SwiGLU-OAI + MXFP8 quant, I 6144 (gfx950)

## What it does

One launch for the SwiGLU-OAI activation of a split-layout gate_up row (`[m, 2 x 6144]` bf16, gate | up, unit column
stride, any row stride) and the MXFP8 quant of the result in the layout the down GEMM (N 6144, K 6144) reads: e4m3
codes row-major `[m, 6144]` or in the decode GEMM's (16, 16) tiles with rows padded to the tile, E8M0 scales
`[ceil(rows / 32) * 32, 192]` in AITER's shuffled layout (padding rows and scale padding unwritten; the GEMM masks
them). Caller: the M3.1 dense FFN at TP2 (I = 6144 per rank): the DSpark draft's 5 layers and the target's dense
layers 0-2, above 16 rows (1-16 rows run `gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144`). Numerics (`reference.py`):
`form` is the caller's, never chosen by row count. `bf16_chain` = the draft's aten chain (`clamp`, `sigmoid`, `mul`,
`add`, each rounded to bf16); `fp32` = the target's `swiglu_oai_split` (fp32 math, one bf16 rounding); both use
`1 / (1 + exp(-x))` with the accurate exp and an IEEE divide. Then the Triton MXFP8 quantizer on the bf16 value: scale
`clamp(ceil(log2(amax / 448)) + 127, 0, 254)` per 32 columns, e4m3 codes of `act * 2^(127 - scale)`. Codes and scales
equal the reference bit for bit on finite inputs, for every workgroup shape.

## Entry points

- `swiglu_oai_mxfp8`: the down GEMM's codes and scales of a gate_up row block, row-major or in (16, 16) tiles; the
  caller takes the tile from `gemm.mxfp8_common.gemm.prequant_code_tile_rows(m, 6144, 6144)` and runs the down GEMM
  with `gemm.mxfp8_common.gemm.mxfp8_gemm_prequant`. The output buffers may be the caller's.
- `warmup_swiglu_oai_mxfp8`: compiles all 8 variants (2 workgroup shapes x 2 forms x 2 code layouts; the row count
  is a runtime argument). Call it before serving or graph capture.

`host.SHAPES` picks the workgroup shape (pieces per row, threads) by row count.

## Performance

One MI355X, graph replay, 100 distinct inputs per graph, us per call, against what vLLM runs at TP2 without it: the
aten chain + quantizer (`bf16_chain`) or Triton `swiglu_oai_split` + quantizer (`fp32`). HBM floor at 6.5 TB/s.

| Rows | Code tile (0: row-major) | bf16_chain baseline | bf16_chain | fp32 baseline | fp32 | HBM floor | fraction (fp32) |
|---|---|---|---|---|---|---|---|
| 7 | 16 | 17.79 | 2.49 | 3.76 | 2.38 | 0.03 | 1% |
| 64 | 64 | 23.51 | 2.78 | 4.31 | 2.70 | 0.30 | 11% |
| 128 | 64 | 27.90 | 3.66 | 5.04 | 3.51 | 0.61 | 17% |
| 256 | 128 | 33.25 | 4.60 | 7.03 | 4.33 | 1.22 | 28% |
| 512 | 128 | 40.20 | 6.74 | 10.40 | 6.20 | 2.43 | 39% |
| 1024 | 0 | 65.17 | 10.64 | 15.40 | 9.71 | 4.87 | 50% |
| 1792 | 0 | 99.54 | 16.90 | 24.12 | 15.06 | 8.52 | 57% |

## Tests

```bash
python3 -m pytest tests/activation/swiglu_oai_mxfp8_i6144 -q
python3 benchmarks/activation/swiglu_oai_mxfp8_i6144/benchmark_swiglu_oai_mxfp8.py [--form fp32] [--sweep]
```
