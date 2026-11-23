# MXFP8 GEMM, N 1792 x K 5120 (fused wq_a + wkv)

## What it does
`out [M, 1792] = x [M, 5120] @ W^T` for the fused wq_a + wkv of the DEP8 decode step, M = the padded decode rows 16-768 (more rows run
the common dispatcher). `x` is bf16, quantized per 1 x 32 block (the vLLM Triton quantizer's bytes) in the GEMM's
registers at small M and by a quantizer launch above it, or pre-quantized codes + row-major E8M0 scales; `W` is the
checkpoint's E4M3 codes with 32 x 32 block scales, expanded per row and preshuffled at load. fp32 sums, one bf16
rounding (`reference.py`: cosine >= 0.99999, L2 <= 1e-3, bitwise repeats; no atomics into outputs, the split-K sum is
fixed-order). Producer: the mHC attention seam (bf16 hidden [M, 5120]). Consumer: `norm/rmsnorm_dual_mxfp8_q1280_kv512` (bf16 [M, 1792]).

## Entry points
- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._ROUTES` (`linear`, `covers`).
- `gemm.mxfp8_n1792_k5120.mxfp8_linear`: the GEMM from a bf16 activation; fresh bf16 output, graph-capture safe, `serves(m)` tells where it beat the served path.
- `gemm.mxfp8_n1792_k5120.preshuffle_weight`: the load-time weight and scale preshuffle (32 x 32 block scales accepted).
- `gemm.mxfp8_n1792_k5120.warmup_mxfp8_linear`: compiles every row and creates the split-K counters before graph capture.
- `kernel.py`: the row table (kernel, tile, K-waves, split-K, `serves`); the kernels are `gemm/mxfp8_decode_prequant` and `gemm/mxfp8_common`.

## Performance
One MI355X, graph replay, random weights in cold copies (> 400 MB), 100 distinct inputs, 9 rounds A B B A, median us.
Served path: vLLM Triton quantizer + AITER `gemm_a8w8_blockscale_bpreshuffle` (installed 0.1.29); `+#6605`: the same
with AITER PR #6605.

| M | us | aiter us | +#6605 us | vs aiter |
|---|---|---|---|---|
| 16 | 5.78 | 7.01 | 7.01 | 1.21x |
| 32 | 6.85 | 7.55 | 7.57 | 1.10x |
| 64 | 8.02 | 8.73 | 8.73 | 1.09x |
| 128 | 9.97 | 10.82 | 10.80 | 1.08x |
| 192 | 11.73 | 12.57 | 12.59 | 1.07x |
| 256 | 13.17 | 14.04 | 14.00 | 1.07x |
| 384 | 15.53 | 15.88 | 15.88 | 1.02x |
| 512 | 18.23 | 19.68 | 19.60 | 1.08x |
| 768 | 22.53 | 20.71 | 20.75 | 0.92x |

Falls back to the served path (AITER) at M = 384, 768 (and rows between them).

Limit: at M 16 the HBM weight stream, 3.19 us for 9.5 MB; achieved 5.78 us (55%). At M 768 fp8 MFMA at the
power-capped clock (3.46 PF measured with 256 CUs busy), then the per-kernel latency chain; achieved 0.62 PF (18%;
AITER on codes 0.79 PF, hipBLASLt fp8 0.78 PF).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n1792_k5120 -q`
- `python3 benchmarks/gemm/mxfp8_n1792_k5120/benchmark_mxfp8_n1792_k5120.py [--ms 16,64] [--scenarios linear] [--out result.json]`
