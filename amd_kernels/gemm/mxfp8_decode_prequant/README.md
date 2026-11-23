# MXFP8 decode GEMM on pre-quantized rows, with a swiglu + MXFP8 epilogue (gfx950)

## What it does
The decode-row MXFP8 GEMM of `gemm/mxfp8_common` (`mxfp8_gemm_decode`: W and A straight from global memory into
the MFMA operands, no LDS staging, K-waves summed in LDS) with three additions. `A_RM`: the activation as E4M3
codes [M, K] and row-major E8M0 scales [M, K / 32], what the norm, wo_a and swiglu kernels emit (vLLM's
`QuantizedActivation`), read as it is. `EPI_LIMIT`: for a gate_up weight [gate; up] the epilogue rounds the sums to
bf16, applies `bf16(bf16(silu(min(gate, limit))) * clamp(up, +-limit))` in fp32 and writes the E4M3 codes
[M, I] and E8M0 bytes [M, I / 32] the down GEMM reads; bit for bit the torch epilogue on the kernel's own bf16
sums. `SK_FUSE`: the split-K sum by the last workgroup of a tile (partials in a slab at agent scope, a ticket on an
epoch counter, summed in split order: deterministic, no atomics into outputs) instead of a reduce launch. W is the
(16, 16) preshuffled weight with AITER-shuffled per-row scales. `reference.py`: exact dequant, fp32 sums, one
rounding; cosine >= 0.99999, L2 <= 1e-3 (epilogue: 5e-3 on the dequantized codes), bitwise repeats.

## Entry points
- `gemm`: the bf16 GEMM from row-major codes + scales, a bf16 activation (quantized in registers) or the common
  quantizer's shuffled layout; `Config` names the row (tiles, K-waves, split-K, scheduling, weight cache modifier,
  `fuse`). `workspace`: the counters and slab of the fused split-K rows, one pair per device and key; one launch at a
  time per pair, created on first use (call a fused row before graph capture).
- `gemm_swiglu_mxfp8`: the same with the epilogue; codes and scales in caller buffers or fresh.
- The per-shape row tables and entries are the `gemm/mxfp8_n<N>_k<K>` packages.

## Performance
One MI355X, graph replay, N 4608 x K 5120 in cold weight copies, us per call
(`benchmarks/gemm/mxfp8_decode_prequant/benchmark_decode_prequant.py`). The AITER GEMM on the same codes is
6.8 / 8.1 / 10.5 / 12.4 us at 16 / 32 / 64 / 128 rows: this kernel ties below 32 rows and loses above, so the shape
packages use it for pre-quantized rows only where it won (`serves`).

| rows | rm | bf16 in | swiglu rm | swiglu bf16 | split-K sum in a launch | in the last workgroup |
|---|---|---|---|---|---|---|
| 16 | 7.61 | 12.47 | 8.96 | 9.24 | 8.54 | 9.10 |
| 32 | 9.09 | 17.15 | 11.07 | 11.92 | 10.63 | 10.88 |
| 64 | 12.04 | 23.31 | 15.71 | 17.04 | 13.55 | 15.19 |
| 128 | 18.42 | 43.47 | 23.82 | 24.94 | 21.70 | 22.63 |

The epilogue form with a bf16 activation takes 9.2 / 11.9 us at 16 / 32 rows against 11.0 / 12.8 us for the served
chain (quantizer, AITER GEMM, clamped SiLU-mul launch); from 64 rows the chain is as fast or faster. The
last-workgroup split-K sum is 0.3-1.6 us slower than the reduce launch for this shape and 0-10% faster for the fused
wq_a + wkv rows: it exists to allow split-K under the epilogue.

Limit: at 16 rows the HBM weight stream, 5.18 us for 24.3 MB; achieved 7.61 us (68%). At 128 rows 0.13 PF of the 4.95 PF
scaled-MFMA ceiling (3%): decode tiles reload W per row tile, so they are not MFMA-bound.

## Tests
- `python3 -m pytest tests/gemm/mxfp8_decode_prequant -q`
- `python3 benchmarks/gemm/mxfp8_decode_prequant/benchmark_decode_prequant.py`
