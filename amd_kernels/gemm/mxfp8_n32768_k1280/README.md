# MXFP8 GEMM, N 32768 x K 1280 (wq_b)

## What it does
`out [M, 32768] = x [M, 1280] @ W^T` for the wq_b of the DEP8 decode step, M = the padded decode rows 16-768 (more rows run
the common dispatcher). `x` is bf16, quantized per 1 x 32 block (the vLLM Triton quantizer's bytes) in the GEMM's
registers at small M and by a quantizer launch above it, or pre-quantized codes + row-major E8M0 scales; `W` is the
checkpoint's E4M3 codes with 32 x 32 block scales, expanded per row and preshuffled at load. fp32 sums, one bf16
rounding (`reference.py`: cosine >= 0.99999, L2 <= 1e-3, bitwise repeats; no atomics into outputs, the split-K sum is
fixed-order). Producer: `norm/rmsnorm_dual_mxfp8_q1280_kv512` (q codes + scales, or bf16). Consumer: the q norm / RoPE / fp8 insert (bf16 [M, 32768]).

## Entry points
- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._SHAPE_HOSTS` (`linear`, `covers`).
- `gemm.mxfp8_n32768_k1280.mxfp8_linear`: the GEMM from a bf16 activation; fresh bf16 output, graph-capture safe, `serves(m)` tells where it beat the served path.
- `gemm.mxfp8_n32768_k1280.preshuffle_weight`: the load-time weight and scale preshuffle (32 x 32 block scales accepted).
- `gemm.mxfp8_n32768_k1280.warmup_mxfp8_linear`: compiles every row and creates the split-K counters before graph capture.
- `kernel.py`: the row table (kernel, tile, K-waves, split-K, `serves`); the kernels are `gemm/mxfp8_decode_prequant` and `gemm/mxfp8_common`.

## Performance
One MI355X, graph replay, random weights in cold copies (> 400 MB), 100 distinct inputs, 9 rounds A B B A, median us.
Served path: vLLM Triton quantizer + AITER `gemm_a8w8_blockscale_bpreshuffle` (installed 0.1.29); `+#6605`: the same
with AITER PR #6605.

| M | us | aiter us | +#6605 us | vs aiter |
|---|---|---|---|---|
| 16 | 8.54 | 9.96 | 9.95 | 1.17x |
| 32 | 10.45 | 10.41 | 10.38 | 1.00x |
| 64 | 12.51 | 11.71 | 11.75 | 0.94x |
| 128 | 15.02 | 14.12 | 14.17 | 0.94x |
| 192 | 18.74 | 17.50 | 17.30 | 0.93x |
| 256 | 20.95 | 20.38 | 20.36 | 0.97x |
| 384 | 28.76 | 24.37 | 24.18 | 0.85x |
| 512 | 32.87 | 30.23 | 26.49 | 0.92x |
| 768 | 48.34 | 43.82 | 42.40 | 0.91x |

Falls back to the served path (AITER) at M = 32, 64, 128, 192, 256, 384, 512, 768 (and rows between them).

Limit: at M 16 the HBM weight stream, 8.02 us for 43.3 MB; achieved 8.54 us (94%). At M 768 fp8 MFMA at the
power-capped clock (3.46 PF measured with 256 CUs busy), then the per-kernel latency chain; achieved 1.35 PF (39%;
AITER on codes 1.56 PF, hipBLASLt fp8 1.60 PF).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n32768_k1280 -q`
- `python3 benchmarks/gemm/mxfp8_n32768_k1280/benchmark_mxfp8_n32768_k1280.py [--ms 16,64] [--scenarios linear] [--out result.json]`
