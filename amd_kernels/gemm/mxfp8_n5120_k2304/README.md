# MXFP8 GEMM, N 5120 x K 2304 (shared-expert down)

## What it does
`out [M, 5120] = x [M, 2304] @ W^T` for the shared-expert down of the DEP8 decode step, M = the padded decode rows 16-768 (more rows run
the common dispatcher). `x` is bf16, quantized per 1 x 32 block (the vLLM Triton quantizer's bytes) in the GEMM's
registers at small M and by a quantizer launch above it, or pre-quantized codes + row-major E8M0 scales; `W` is the
checkpoint's E4M3 codes with 32 x 32 block scales, expanded per row and preshuffled at load. fp32 sums, one bf16
rounding (`reference.py`: cosine >= 0.99999, L2 <= 1e-3, bitwise repeats; no atomics into outputs, the split-K sum is
fixed-order). Producer: `activation/clamp_silu_mul_mxfp8_i2304` or the gate_up epilogue (codes [M, 2304], scales [M, 72]). Consumer: the shared-expert add into the MoE output (bf16 [M, 5120]).

## Entry points
- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._ROUTES` (`linear`, `gemm` (codes), `covers`).
- `gemm.mxfp8_n5120_k2304.mxfp8_linear`: the GEMM from a bf16 activation; fresh bf16 output, graph-capture safe, `serves(m)` tells where it beat the served path.
- `gemm.mxfp8_n5120_k2304.preshuffle_weight`: the load-time weight and scale preshuffle (32 x 32 block scales accepted).
- `gemm.mxfp8_n5120_k2304.warmup_mxfp8_linear`: compiles every row and creates the split-K counters before graph capture.
- `kernel.py`: the row table (kernel, tile, K-waves, split-K, `serves`); the kernels are `gemm/mxfp8_decode_prequant` and `gemm/mxfp8_common`.

## Performance
One MI355X, graph replay, random weights in cold copies (> 400 MB), 100 distinct inputs, 9 rounds A B B A, median us.
Served path: vLLM Triton quantizer + AITER `gemm_a8w8_blockscale_bpreshuffle` (installed 0.1.29); `+#6605`: the same
with AITER PR #6605.

| M | us | aiter us | +#6605 us | vs aiter |
|---|---|---|---|---|
| 16 | 5.43 | 6.17 | 6.13 | 1.14x |
| 32 | 6.65 | 6.89 | 6.85 | 1.04x |
| 64 | 7.37 | 8.34 | 8.33 | 1.13x |
| 128 | 9.27 | 9.41 | 9.39 | 1.02x |
| 192 | 10.22 | 10.58 | 10.59 | 1.04x |
| 256 | 11.56 | 11.90 | 11.88 | 1.03x |
| 384 | 12.94 | 13.87 | 13.93 | 1.07x |
| 512 | 15.44 | 15.17 | 15.19 | 0.98x |
| 768 | 19.14 | 18.71 | 18.49 | 0.98x |

Falls back to the served path (AITER) at M = 128, 256, 512, 768 (and rows between them).

Limit: at M 16 the HBM weight stream, 3.53 us for 12.2 MB; achieved 5.43 us (65%). At M 768 fp8 MFMA at the
power-capped clock (3.46 PF measured with 256 CUs busy), then the per-kernel latency chain; achieved 0.93 PF (27%;
AITER on codes 1.12 PF, hipBLASLt fp8 1.11 PF).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n5120_k2304 -q`
- `python3 benchmarks/gemm/mxfp8_n5120_k2304/benchmark_mxfp8_n5120_k2304.py [--ms 16,64] [--scenarios linear] [--out result.json]`
