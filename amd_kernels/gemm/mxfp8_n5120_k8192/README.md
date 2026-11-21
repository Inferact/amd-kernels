# MXFP8 GEMM, N 5120 x K 8192 (wo_b)

## What it does
`out [M, 5120] = x [M, 8192] @ W^T` for the wo_b of the DEP8 decode step, M = the padded decode rows 16-768 (more rows run
the common dispatcher). `x` is bf16, quantized per 1 x 32 block (the vLLM Triton quantizer's bytes) in the GEMM's
registers at small M and by a quantizer launch above it, or pre-quantized codes + row-major E8M0 scales; `W` is the
checkpoint's E4M3 codes with 32 x 32 block scales, expanded per row and preshuffled at load. fp32 sums, one bf16
rounding (`reference.py`: cosine >= 0.99999, L2 <= 1e-3, bitwise repeats; no atomics into outputs, the split-K sum is
fixed-order). Producer: `gemm/mxfp8_grouped_g8_n1024_k4096` MXFP8 epilogue, or bf16. Consumer: the mHC post / residual seam (bf16 [M, 5120]).

## Entry points
- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._SHAPE_HOSTS` (`linear`, `covers`).
- `gemm.mxfp8_n5120_k8192.mxfp8_linear`: the GEMM from a bf16 activation; fresh bf16 output, graph-capture safe, `serves(m)` tells where it beat the served path.
- `gemm.mxfp8_n5120_k8192.preshuffle_weight`: the load-time weight and scale preshuffle (32 x 32 block scales accepted).
- `gemm.mxfp8_n5120_k8192.warmup_mxfp8_linear`: compiles every row and creates the split-K counters before graph capture.
- `kernel.py`: the row table (kernel, tile, K-waves, split-K, `serves`); the kernels are `gemm/mxfp8_decode_prequant` and `gemm/mxfp8_common`.

## Performance
One MI355X, graph replay, random weights in cold copies (> 400 MB), 100 distinct inputs, 9 rounds A B B A, median us.
Served path: vLLM Triton quantizer + AITER `gemm_a8w8_blockscale_bpreshuffle` (installed 0.1.29); `+#6605`: the same
with AITER PR #6605.

| M | us | aiter us | +#6605 us | vs aiter |
|---|---|---|---|---|
| 16 | 11.72 | 11.59 | 11.53 | 0.99x |
| 32 | 13.72 | 13.26 | 13.20 | 0.97x |
| 64 | 17.24 | 15.99 | 15.92 | 0.93x |
| 128 | 21.32 | 20.70 | 20.71 | 0.97x |
| 192 | 23.32 | 23.19 | 23.23 | 0.99x |
| 256 | 25.86 | 29.01 | 29.03 | 1.12x |
| 384 | 30.08 | 34.19 | 34.19 | 1.14x |
| 512 | 33.54 | 39.36 | 39.09 | 1.17x |
| 768 | 47.69 | 49.00 | 48.98 | 1.03x |

Falls back to the served path (AITER) at M = 16, 32, 64, 128, 192, 768 (and rows between them).

Limit: at M 16 the HBM weight stream, 8.02 us for 43.3 MB; achieved 11.72 us (68%). At M 768 fp8 MFMA at the
power-capped clock (3.46 PF measured with 256 CUs busy), then the per-kernel latency chain; achieved 1.33 PF (38%;
AITER on codes 1.41 PF, hipBLASLt fp8 1.47 PF).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n5120_k8192 -q`
- `python3 benchmarks/gemm/mxfp8_n5120_k8192/benchmark_mxfp8_n5120_k8192.py [--ms 16,64] [--scenarios linear] [--out result.json]`
