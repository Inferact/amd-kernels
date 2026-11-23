# MXFP8 GEMM, N 4608 x K 5120 (shared-expert gate_up)

## What it does
`out [M, 4608] = x [M, 5120] @ W^T` for the shared-expert gate_up of the DEP8 decode step, M = the padded decode rows 16-768 (more rows run
the common dispatcher). `x` is bf16, quantized per 1 x 32 block (the vLLM Triton quantizer's bytes) in the GEMM's
registers at small M and by a quantizer launch above it, or pre-quantized codes + row-major E8M0 scales; `W` is the
checkpoint's E4M3 codes with 32 x 32 block scales, expanded per row and preshuffled at load. fp32 sums, one bf16
rounding (`reference.py`: cosine >= 0.99999, L2 <= 1e-3, bitwise repeats; no atomics into outputs, the split-K sum is
fixed-order). Producer: the mHC ffn seam (bf16 [M, 5120], or its MXFP8 output). Consumer: `gate_up_swiglu_mxfp8`: the down GEMM below (codes [M, 2304], scales [M, 72]).

## Entry points
- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._ROUTES` (`linear`, `covers`).
- `gemm.mxfp8_n4608_k5120.mxfp8_linear`: the GEMM from a bf16 activation; fresh bf16 output, graph-capture safe, `serves(m)` tells where it beat the served path.
- `gemm.mxfp8_n4608_k5120.gate_up_swiglu_mxfp8` / `gate_up_swiglu_mxfp8_prequant`: the GEMM with clamped SiLU-mul and MXFP8 quantization in the epilogue (bf16 / MXFP8 in; codes [M, 2304] + scales [M, 72] out).
- `gemm.mxfp8_n4608_k5120.preshuffle_weight`: the load-time weight and scale preshuffle (32 x 32 block scales accepted).
- `gemm.mxfp8_n4608_k5120.warmup_mxfp8_linear`: compiles every row and creates the split-K counters before graph capture.
- `kernel.py`: the row table (kernel, tile, K-waves, split-K, `serves`); the kernels are `gemm/mxfp8_decode_prequant` and `gemm/mxfp8_common`.

## Performance
One MI355X, graph replay, random weights in cold copies (> 400 MB), 100 distinct inputs, 9 rounds A B B A, median us.
Served path: vLLM Triton quantizer + AITER `gemm_a8w8_blockscale_bpreshuffle` (installed 0.1.29); `+#6605`: the same
with AITER PR #6605.

| M | us | aiter us | +#6605 us | vs aiter |
|---|---|---|---|---|
| 16 | 8.37 | 8.59 | 8.56 | 1.03x |
| 32 | 10.71 | 10.25 | 10.20 | 0.96x |
| 64 | 12.38 | 12.59 | 12.59 | 1.02x |
| 128 | 15.37 | 14.35 | 14.32 | 0.93x |
| 192 | 16.84 | 17.92 | 18.22 | 1.06x |
| 256 | 19.53 | 20.55 | 20.55 | 1.05x |
| 384 | 21.65 | 23.55 | 23.54 | 1.09x |
| 512 | 26.30 | 27.06 | 27.09 | 1.03x |
| 768 | 32.18 | 32.21 | 32.15 | 1.00x |

Falls back to the served path (AITER) at M = 16, 32, 64, 128, 512, 768 (and rows between them).

Other forms, us ours / served chain (AITER GEMM + the clamped SiLU-mul launch) at M 16 / 64 / 192 / 384 / 768: swiglu
9.5 / 11.0, 15.0 / 15.3, 19.6 / 20.6, 27.1 / 26.5, 46.3 / 36.4; swiglu_prequant 8.8 / 9.1, 13.7 / 13.0, 22.1 / 18.0,
37.5 / 23.3, 91.8 / 33.4.

Limit: at M 16 the HBM weight stream, 5.18 us for 24.3 MB; achieved 8.37 us (62%). At M 768 fp8 MFMA at the
power-capped clock (3.46 PF measured with 256 CUs busy), then the per-kernel latency chain; achieved 1.12 PF (32%;
AITER on codes 1.24 PF, hipBLASLt fp8 1.29 PF).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n4608_k5120 -q`
- `python3 benchmarks/gemm/mxfp8_n4608_k5120/benchmark_mxfp8_n4608_k5120.py [--ms 16,64] [--scenarios linear] [--out result.json]`
