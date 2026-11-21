# bf16 latent down projection + MXFP4 quant, K 7168 x N 3584 (the routed experts' input)

## What it does
`codes, scales = mxfp4(bf16(x [M, 7168] @ W^T))` for the MoE layer's replicated latent `routed_expert_down_proj`
(bf16 W [3584, 7168], 51 MB, M 1-192; above 192 rows launches of 192). bf16 operands, fp32 sums (K split in 4 slices
summed in a fixed order, no output atomics, bitwise repeatable), one bf16 rounding, then the quant the aiter a4w4 MoE
takes: per row, 32 columns share an e8m0 scale `ceil(log2(amax / 6))` floored at 92, RNE e2m1 codes saturating at 6.
Output layout = `per_1x32_mx_quant_hip(y, quant_dtype=fp4x2, shuffle=False)`: codes `[M, 1792]` fp4x2, scales
`[M, 112]` e8m0, row-major (what `mxfp4_moe_sort_fwd` / `prequantized_a1` consume); optionally the bf16 `y` too.
Contract (`reference.py`): the codes and scales of the kernel's own bf16 `y` equal `quant_reference` and aiter's quant
bit for bit; `y` differs from the fp64 product only by sum order (cosine >= 0.99999). Weight layout `row` (the
checkpoint's) or `shuf` (bit-exact permutation once at load, one 1 KB DMA per fragment).

## Entry points
- `bf16_latent_down_mxfp4_quant_k7168_n3584`: the projection and quant of any row count, graph-safe, no host sync; an
  `amd_kernels.warm_up` entry (the first call of each mode, layout x bf16 output, compiles its 12 row tiles; the first
  call on a device also runs a placement probe, `xcc_probe`). One call at a time per stream (partial slabs and tile
  counters live in a per-stream workspace).
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout.
- `supports`: whether a weight is the bf16 contiguous [3584, 7168] one and there is no bias.

`host.py` picks the row tile (16 rows per fragment) and the K split (`KS_BY_MT`).

## Performance
One MI355X, exclusive GPU, graph replay of 12 calls over 12 distinct weight copies (617 MB > the MALL) and
activation sets, min of 15 rounds, us. Stock = vLLM `rocm_unquantized_gemm_impl` (VLLM_ROCM_USE_AITER=1: aiter tuned
hgemm) + aiter `per_1x32_mx_quant_hip`. Floor = the 51.4 MB weight stream at the measured nontemporal HBM rate:
1.57 us graph node + 7.4 us (6.95 TB/s) = 9.0 us. DMA-only = the same grid issuing the same weight + activation DMAs
with no MFMA and no epilogue.

| M | 1 | 4 | 12 | 24 | 40 | 65 | 80 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|---|
| stock gemm + quant | 11.4 | 12.6 | 14.3 | 14.7 | 15.6 | 19.7 | 20.2 | 23.3 | 33.6 |
| fused | 11.1 | 10.8 | 10.9 | 11.5 | 12.1 | 13.9 | 14.4 | 17.2 | 23.1 |
| DMA-only | 8.9 | 8.8 | 8.9 | 9.2 | 9.7 | 10.8 | 11.4 | 13.6 | 17.7 |
| % of the 9.0 us floor | 81 | 83 | 82 | 78 | 74 | 65 | 62 | 52 | 39 |
| speedup | 1.03x | 1.17x | 1.31x | 1.27x | 1.28x | 1.42x | 1.40x | 1.35x | 1.45x |

The fused kernel runs at 77-83% of its own DMA-only skeleton at every M: the weight stream is at the HBM ceiling,
and from M 40 the per-CU activation path (L2 to LDS, ~110 GB/s per CU, additive to the weight stream) sets the floor.
The epilogue after the slowest main loop is 1.2 us at M 12 (K halves 0.16, slab stores 0.25, wait for the tile 0.45,
slab loads + LDS tile 0.45, quant + stores 0.3) and 3.6 us at M 192.

## Tests
- `python3 -m pytest tests/gemm/bf16_latent_down_mxfp4_quant_k7168_n3584 -q`
- `python3 benchmarks/gemm/bf16_latent_down_mxfp4_quant_k7168_n3584/benchmark_down_quant.py [M ...]`
