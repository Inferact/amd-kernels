# Grouped MXFP8 GEMM, 8 groups x N 1024 x K 4096 (low-rank attention output projection)

## What it does
`out[m, g, :] = x[m, g, :] @ W[g]^T` for 8 groups: the first (grouped) GEMM of DeepSeek-V4.1-Flash's low-rank attention
output projection at DP attention (TP1), 1-1024 decode / chunk rows. `x`: E4M3 codes [M, 8 * 4096] with row-major E8M0
1 x 32 scales [M, 8 * 128] (the sparse-MLA epilogue's output as is); `W`: the checkpoint's E4M3 codes [8 * 1024, 4096]
in AITER's (16, 16) preshuffle with its 32 x 32 E8M0 scales [8, 32, 128]. `grouped_gemm` returns bf16 [M, 8 * 1024];
`grouped_gemm_mxfp8` rounds the tile to bf16 in LDS and returns its MXFP8 quantization, E4M3 codes [M, 8 * 1024] and
row-major E8M0 scales [M, 8 * 32] (uint8), what the next GEMM (`[5120, 8192]`) takes as a pre-quantized activation:
no quantize launch, no bf16 round trip. A group's 1024 columns are 32 whole quant blocks, so each block is local to
one workgroup. `reference.py`: exact dequant, fp32 sums per group, one bf16 rounding; cos >= 0.99999, L2 <= 1e-3,
bitwise repeats; the MXFP8 entry is `quantize_mxfp8` of that bf16 (bytes equal at the same accumulation order, else
cos >= 0.99999, L2 <= 5e-3). No atomics into outputs; split-K, where used, sums partials in split order.

## Entry points
- `grouped_gemm`: bf16 out, or into a caller-owned contiguous `out`.
- `grouped_gemm_mxfp8`: the MXFP8 epilogue; the result goes into vLLM's `QuantizedActivation(data, scale)`.
- `warmup_grouped_gemm`: compiles every row window of both entries and allocates the split-K workspace; call it
  before graph capture (a first call inside capture does not warm up).
- `host.select`: the config of a row count (`host._ROWS`); `kernel.Config`: tile, waves, stages, split K, B path.

The kernel is the AITER FlyDSL batched GEMM (`bmm_a8w8_mxscale_gfx950`, tile / pipeline unchanged) specialized to this
shape with the epilogue added; one workspace per device, so launches go on one stream.

## Performance
One MI355X. Graph replay, 16 cold weight copies (> 500 MB), 100 distinct inputs, A B B A rounds, median µs per call.
Baseline: AITER `batched_gemm_a8w8_mxscale_bpreshuffle` (its tuned FlyDSL rows), and for the MXFP8 entry that plus
vLLM's `mxfp8_e4m3_quantize` launch. Ideal: `max(bytes / 8 TB/s, flops / 5.03 PF)`, 33.6 MB of weight.

| M | bf16 out µs | AITER µs | MXFP8 out µs | AITER + quant µs | ideal µs |
|---|---|---|---|---|---|
| 1 | 6.67 | 6.52 | 6.81 | 8.56 | 4.20 |
| 16 | 7.17 | 7.01 | 7.35 | 9.11 | 4.30 |
| 32 | 8.12 | 9.41 | 8.29 | 11.34 | 4.40 |
| 64 | 9.39 | 9.95 | 9.52 | 12.16 | 4.60 |
| 128 | 12.14 | 12.45 | 12.17 | 14.60 | 5.00 |
| 192 | 14.64 | 14.43 | 14.78 | 17.49 | 5.40 |
| 256 | 16.21 | 16.67 | 16.37 | 18.92 | 5.80 |
| 384 | 19.68 | 20.73 | 19.77 | 23.21 | 6.61 |
| 512 | 23.18 | 23.00 | 23.38 | 25.89 | 7.41 |
| 768 | 31.47 | 31.01 | 31.40 | 34.52 | 10.25 |
| 1024 | 37.40 | 36.36 | 37.17 | 41.13 | 13.66 |

The GEMM runs at AITER's speed (AITER PR #6605 changes the dense 8-wave epilogue, not this batched kernel, and times
the same). The MXFP8 epilogue costs nothing measurable, so it takes 1.7-3.8 us (9-22 %) off AITER + quantize at every M.
Limit: at M 1-16 the HBM weight stream, 6.7 us for 33.5 MB; achieved 7.2 us (93 %). At M 768 the global -> LDS / VGPR
operand streams (the kernel without MFMAs: 18.3 us; achieved 31.4 us, 58 %), scaled MFMA 1.64 of 4.2 PF (39 %).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_grouped_g8_n1024_k4096 -q`
- `python3 benchmarks/gemm/mxfp8_grouped_g8_n1024_k4096/benchmark_grouped_gemm.py [--rows 16,64,...]`
- `python3 benchmarks/gemm/mxfp8_grouped_g8_n1024_k4096/benchmark_config_sweep.py --rows 16,192 --random 200` (the sweep)
