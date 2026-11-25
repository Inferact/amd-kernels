# MXFP8 GEMM, N 4096 x K 1280 (indexer wq_b)

## What it does
`out [M, 4096] = x [M, 1280] @ W^T` for the indexer wq_b of the DEP8 decode step, M = the padded decode rows 16-768 (more rows run
the common dispatcher). `x` is bf16, quantized per 1 x 32 block (the vLLM Triton quantizer's bytes) in the GEMM's
registers at small M and by a quantizer launch above it, or pre-quantized codes + row-major E8M0 scales; `W` is the
checkpoint's E4M3 codes with 32 x 32 block scales, expanded per row and preshuffled at load. fp32 sums, one bf16
rounding (`reference.py`: cosine >= 0.99999, L2 <= 1e-3, bitwise repeats; no atomics into outputs, the split-K sum is
fixed-order). Producer: `norm/rmsnorm_dual_mxfp8_q1280_kv512` (q codes + scales, or bf16). Consumer: `indexer/q_rope_mxfp4_quant_h32_d128_rot64` (bf16 [M, 4096]).

## Entry points
- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._ROUTES` (`linear`, `gemm` (codes), `covers`).
- `gemm.mxfp8_n4096_k1280.mxfp8_linear`: the GEMM from a bf16 activation; fresh bf16 output, graph-capture safe, `serves(m)` tells where it beat the served path.
- `gemm.mxfp8_n4096_k1280.mxfp8_linear_prequant`: the same from E4M3 codes + row-major E8M0 scales (`QuantizedActivation`).
- `gemm.mxfp8_n4096_k1280.preshuffle_weight`: the load-time weight and scale preshuffle (32 x 32 block scales accepted).
- `gemm.mxfp8_n4096_k1280.warmup_mxfp8_linear`: compiles every row and creates the split-K counters before graph capture.
- `kernel.py`: the row table (kernel, tile, K-waves, split-K, `serves`); the kernels are `gemm/mxfp8_decode_prequant` and `gemm/mxfp8_common`.

## Performance
One MI355X, graph replay, random weights in cold copies (> 400 MB), 100 distinct inputs, 9 rounds A B B A, median us.
Served path: vLLM Triton quantizer + AITER `gemm_a8w8_blockscale_bpreshuffle` (installed 0.1.29); `+#6605`: the same
with AITER PR #6605.

| M | us | aiter us | +#6605 us | vs aiter |
|---|---|---|---|---|
| 16 | 3.85 | 5.06 | 4.99 | 1.31x |
| 32 | 4.07 | 5.39 | 5.30 | 1.33x |
| 64 | 4.84 | 6.19 | 6.19 | 1.28x |
| 128 | 6.19 | 6.71 | 6.68 | 1.08x |
| 192 | 7.52 | 7.43 | 7.42 | 0.99x |
| 256 | 8.01 | 7.84 | 7.79 | 0.98x |
| 384 | 9.64 | 9.04 | 9.06 | 0.94x |
| 512 | 11.08 | 9.96 | 9.99 | 0.90x |
| 768 | 12.81 | 11.88 | 11.88 | 0.93x |

Falls back to the served path (AITER) at M = 192, 256, 384, 512, 768 (and rows between them).

Other forms, us ours / served chain at M 16 / 64 / 192 / 384 / 768: prequant 3.1 / 3.1, 3.7 / 4.1, 5.3 / 5.2,
8.2 / 6.7, 14.3 / 9.1.

Limit: at M 16 the HBM weight stream, 2.49 us for 5.4 MB; achieved 3.85 us (65%). At M 768 fp8 MFMA at the
power-capped clock (3.46 PF measured with 256 CUs busy), then the per-kernel latency chain; achieved 0.63 PF (18%;
AITER on codes 0.95 PF, hipBLASLt fp8 0.81 PF).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_n4096_k1280 -q`
- `python3 benchmarks/gemm/mxfp8_n4096_k1280/benchmark_mxfp8_n4096_k1280.py [--ms 16,64] [--scenarios linear] [--out result.json]`
