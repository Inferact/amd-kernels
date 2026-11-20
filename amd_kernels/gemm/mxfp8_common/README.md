# MXFP8 GEMM: the dispatcher and the shared kernels (gfx950)

## What it does
`out [M, N] = x [M, K] @ W^T` for the MXFP8 dense linears of MiniMax-M3.1-Flash. W: E4M3 codes [N, K] + E8M0 per-32
scales [N, K / 32], preshuffled at load ((16, 16) codes, AITER `shuffle_scale_a16w4` scales). x: bf16 / fp16 [M, K],
K % 256 == 0, quantized per 1 x 32 block. Shapes (N x K) of the TP4 / TP2 shards: QKV 2560 / 4992 x 6144, o_proj
6144 x 2048 / 4096, `o_proj_full` 6144 x 8192, dense gate_up 6144 / 12288 x 6144, dense down 6144 x 3072 / 6144,
shared-expert gate_up 1536 / 3072 x 6144 and down 6144 x 768 / 1536, draft QKV 2304 / 4608 x 6144, draft context K / V
1280 / 2560 x 6144. Callers: vLLM's MXFP8 linears, the norm + quant producers, the QKV chain, SP hand-offs.
`reference.py`: exact dequant, fp32 sums, one rounding; cos >= 0.99999, L2 <= 1e-3, bitwise repeats.

## Entry points
- `gemm.preshuffle_mxfp8_weight`, `gemm.preshuffle_mxfp8_weight_`: the load-time weight preshuffle; the in-place form
  returns False off `tuned_shapes()`.
- `gemm.tuned_shapes`: the (N, K) with tuned rows (none for fp16 output); switch only these shapes.
- `gemm.mxfp8_linear`: the GEMM from a bf16 / fp16 activation, quantized in the selected row's code layout.
- `gemm.mxfp8_gemm`: the GEMM from row-major activation codes (never a decode row); row-major scales are shuffled
  first (torch).
- `gemm.prequant_code_tile_rows`: for a code producer, 0 (write row-major codes) or the row tile its (16, 16) codes
  are padded to; `gemm.prequant_buffers`: those buffers.
- `gemm.mxfp8_gemm_shuffled`: the GEMM on (16, 16) codes; it can return `gemm.SkPartials` (fp32 split-K partials)
  for a consumer that sums them.
- `gemm.mxfp8_gemm_prequant`: either form, as `prequant_code_tile_rows` says.
- `gemm.warmup_mxfp8_gemm_weight`: compiles every variant up to a row count, once per (N, K, device, dtype); call it
  before graph capture (a first call does not warm up).
- `gemm.set_dbo_query`: tells the 4-wave rows whether they run inside a DBO microbatch (else `gemm.DBO` answers).
- `gemm.mxfp8_e4m3_quantize_aiter`, `gemm.mxfp8_e4m3_quantize_shuffled`: the activation quantizer, row-major or
  (16, 16) codes.

Dispatch: `gemm._select_config` asks the shape's folder (`gemm/mxfp8_n<N>_k<K>/host.py`: `ROWS`, `select`); a folder's `host.select`
(`gemm._SHAPE_HOSTS`) answers first; the `decode_splitk` rows run `mxfp8_gemm_decode_splitk.py`. `out=` (a caller's
arena) is accepted for o_proj 6144 x 2048 / 4096.

## Performance
Graph replay, cold rotated weight copies (> 400 MB), bf16 entry (quantizer + GEMM), µs per call, median; baseline
vLLM's Triton `tl.dot_scaled` linear (`rocm_native._mxfp8_dot_scaled_linear`, its `_M31_TILES`), the same bf16 input.

| N x K | M 1 | M 8 | M 32 | M 128 | M 256 | M 416 | M 1024 | M 8192 | vs Triton |
|---|---|---|---|---|---|---|---|---|---|
| 3072 x 6144 | 5.18 | 6.53 | 8.98 | 12.22 | 16.03 | 21.15 | 36.69 | 140.2 | -30..-49% |
| 6144 x 1536 | 3.51 | 3.79 | 5.82 | 8.09 | 9.90 | 12.39 | 19.19 | 76.7 | -15..-46% |
| 4608 x 6144 | 6.57 | 8.02 | 10.93 | 14.24 | 19.18 | 26.13 | 40.08 | 194.2 | -38..-58% |
| 12288 x 6144 | 12.96 | 14.65 | 17.62 | 26.20 | 32.77 | 49.40 | 75.99 | 456.9 | -28..-59% |
| 6144 x 6144 | 7.55 | 8.65 | 12.03 | 15.52 | 21.32 | 28.74 | 47.34 | 232.9 | -17..-53% |
| 2560 x 6144 | 4.56 | 6.34 | 9.72 | 13.83 | 16.59 | 20.05 | 27.33 | 126.1 | +1..-59% |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_common -q`
- `python3 benchmarks/gemm/mxfp8_common/benchmark_gemm.py --variant <a test_gemm.CASES name>`
- `python3 benchmarks/gemm/mxfp8_common/benchmark_table.py --shape N,K` (vLLM on the path for the baseline);
  `sweep_rows.py --shape N,K --ms ... --family all` re-tunes a shape's rows; `benchmark_decode_splitk.py --shape N,K
  --ms ... --cands ...` A/Bs the table's row against explicit decode / split-K decode rows.
