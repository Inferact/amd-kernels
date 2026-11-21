# mHC seam finisher: tile records -> gates + RMSNorm (+ MXFP8), hidden 5120

## What it does
The second half of an mHC seam when the producer GEMM has already written the new residual:
`gemm/mxfp8_postmix_hc4_n5120_k8192` and `gemm/mxfp8_postmix_add_hc4_n5120_k2304` (`postmix_gates_gemm`) leave the
collapse `x1` (bf16 [m, 5120]) and per (row, column block, K-wave) a 32-float record: 24 gate-projection sums of
`flatten(R') @ fn^T`, sum R'^2 (slot 24) and sum x1^2 (slot 25). One workgroup of 640 lanes per row loads its records at
once, folds them (DPP / permlane swaps, then the 10 waves in a fixed order through LDS: bitwise repeatable), and splits:
every lane normalizes 8 columns of x1, `bf16(x1 * rsqrt(mean(x1^2) + eps) * w)`, with the optional MXFP8 codes +
row-major E8M0 bytes of the result (the contract of `norm/rmsnorm_dual_mxfp8_q1280_kv512`; the consumers are the wq_a +
wkv GEMM, the router and the shared gate_up), while wave 0 computes pre / post sigmoids and the 4 x 4 Sinkhorn exactly
as `norm/mhc_post_pre_rmsnorm_h5120` does. It never reads the residual (51 KB per row of the seam's traffic). Outputs:
post [m, 4, 1], comb [m, 4, 4], layer_input bf16 [m, 5120], next_pre [m, 4] fp32 (the next producer GEMM's `post_mix` /
`comb_mix` / `pre_mix`), codes / bytes with `quant`; records fp32 [m, nb, 32] (nb <= 640), m <= 4096. `reference.py` is
the contract: gates agree to a few fp32 ulp (records summed in another order), `layer_input` to one bf16 ulp in a few
elements, MXFP8 bytes exact against the quantizer applied to the kernel's own bf16 output.

## Entry points
- `mhc_gate_finish`: the finisher; CUDA-graph safe, deterministic.

An entry (`amd_kernels.warm_up`): one kernel per (nb, quant, Sinkhorn repeats), compiled on its first call; the row
count is a runtime argument, no dispatch.

## Performance
One MI355X (shared), graph replay of 100 distinct inputs, rounds A B B A, us, MXFP8 output. Baseline: seam =
`mhc_post_pre_rmsnorm` without post-mix (reads the residual, projects it itself); nb = records per row (64 and 160 are
the tuned producers').

| rows | seam | nb 64 | nb 160 |
|---:|---:|---:|---:|
| 16 | 6.96 | 3.21 | 3.29 |
| 64 | 7.54 | 3.49 | 3.45 |
| 128 | 7.99 | 3.63 | 4.10 |
| 192 | 9.38 | 4.19 | 4.39 |
| 256 | 9.58 | 4.35 | 4.77 |
| 384 | 12.19 | 5.27 | 5.96 |
| 512 | 15.22 | 5.83 | 6.64 |
| 768 | 19.16 | 7.84 | 9.05 |

At 16 rows the kernel is at 81% of the empty graph node floor (2.6 us; one launch, one HBM round trip); at 384
rows it moves 3.2 TB/s, 60% of the 5.3 TB/s HBM ceiling.

## Tests
- `python3 -m pytest tests/norm/mhc_gate_finish_rmsnorm_h5120 -q`
- `python3 benchmarks/norm/mhc_gate_finish_rmsnorm_h5120/benchmark_finish.py`
