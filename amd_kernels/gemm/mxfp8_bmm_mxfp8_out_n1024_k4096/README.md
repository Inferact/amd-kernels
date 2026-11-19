# Batched MXFP8 GEMM to MXFP8, N 1024 x K 4096 (gfx950)

## What it does
`q, s = quant(bf16(x[:, g] @ w[g]^T))` for every group g: DeepSeek-V4.1's grouped `wo_a` (8 groups of 8 heads x 512
at attention TP1, 4 / 2 / 1 at TP2 / 4 / 8), handing `wo_b` its MXFP8 input without a bf16 round trip and a
separate quantizer. x: e4m3 [M, G x 4096] with row-major E8M0 [M, G x 128] scales (the sparse attention's MXFP8
output); w: e4m3 [G, 1024, 4096] preshuffled (`shuffle_weight(., (16, 16))`) with 32x32 E8M0 block scales [G, 32,
128]. Out: row-major e4m3 [M, G x 1024] and E8M0 [M, G x 32]. Numerics (`reference.py`): the bf16 GEMM output
equals AITER's FlyDSL batched GEMM bit for bit (its main loop), then the Triton MXFP8 quantizer's bytes; equal to
AITER's tuned dispatch (FlyDSL or OPUS) + vLLM's quantizer except on fp32-subnormal outputs, which OPUS flushes.

## Entry points
- `mxfp8_bmm_mxfp8_out`: codes and scales of the GEMM; warms every tile config up on its first call.
- `host.select`: the tile config for a row count (AITER's tuned FlyDSL rows); `host.parse_config` reads an AITER
  kernelName suffix.

## Performance
One MI355X. Graph replay, 8 groups, weights in 8 rotated copies (256 MB), inputs rotated past 512 MB, median µs per
call. Baseline: AITER `batched_gemm_a8w8_mxscale_bpreshuffle` (tuned dispatch, bf16) + vLLM
`_mxfp8_e4m3_quantize_triton`.

| M | AITER GEMM alone | GEMM + quantizer | `mxfp8_bmm_mxfp8_out` | Delta |
|---|---|---|---|---|
| 1 / 16 / 128 | 6.46 / 7.03 / 11.89 | 8.35 / 9.00 / 14.19 | 6.56 / 7.14 / 12.04 | -1.8 / -1.9 / -2.2 |
| 512 / 1024 | 22.08 / 34.62 | 25.97 / 40.39 | 22.08 / 35.45 | -3.9 / -4.9 |
| 2048 / 4096 | 67.49 / 131.08 | 76.98 / 144.35 | 66.08 / 126.37 | -10.9 / -18.0 |
| 8192 / 16384 | 266.71 / 481.89 | 297.75 / 541.91 | 246.77 / 489.18 | -51.0 / -52.7 |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_bmm_mxfp8_out_n1024_k4096 -q`
- `python3 benchmarks/gemm/mxfp8_bmm_mxfp8_out_n1024_k4096/benchmark_mxfp8_bmm_mxfp8_out.py` (needs vLLM)
