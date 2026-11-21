# wo_b MXFP8 GEMM with the mHC post-mix epilogue, N 5120 x K 8192 (gfx950)

## What it does
`gemm/mxfp8_grouped_g8_n1024_k4096` (wo_a) leaves the MXFP8 codes [M, 8192] (E4M3) and row-major E8M0 bytes [M, 256];
this kernel reads them as they are (no shuffle, no standalone quantize or reformat launch) and multiplies the
preshuffled wo_b weight. The decode-row mainloop of `gemm/mxfp8_decode_prequant` (K-waves summed in LDS, optional split K
by the last arriving workgroup, no atomics into outputs) ends in a lane that holds 4 consecutive columns of one row,
so the mHC post-mix of the seam is lane-local: `y = bf16(sum)`, `R'[j] = bf16(p[j] * y + sum_h c[h][j] * R[h])` (fma
order of `norm/mhc_post_pre_rmsnorm_h5120`, so R' equals the separate GEMM followed by that seam's post-mix bit for
bit). y never reaches HBM. `post` [M, 4, 1] and `comb` [M, 4, 4] fp32 are the previous seam's gates.
`postmix_gates_gemm` also writes what the next seam would recompute: the collapse `x1 = bf16(sum_j pre_j R'_j)` with the
carried pre gate, `aux_mean` the mean of the four new streams, and one 128 B record per (row, column block, K-wave) holding the
partial gate projection (the epilogue multiplies the bf16 R' by `fn` split into bf16 hi + lo on the MFMA, fp32
accumulation: 24 mixes, sum R'^2, sum x1^2). `norm/mhc_gate_finish_rmsnorm_h5120` reduces the records, runs Sinkhorn
and the RMSNorm of x1 (+ MXFP8), and its next_pre / post / comb are the next epilogue's gates. `reference.py` is the
contract (cosine >= 0.99999 against torch; the sums over the records agree with the torch projection to 1e-5).

## Entry points
- `postmix_gemm`: GEMM + post-mix; returns the new residual bf16 [M, 4, 5120] (the next seam's `residual`).
- `postmix_gates_gemm`: the same + x1, records (and the mean); feeds `mhc_gate_finish`.
- `pack_gate_weight`, `preshuffle_weight`: once per layer weight; `warmup_postmix_gemm`: compiles every row.
- Dispatch (`host._ROWS`, `host._GATE_ROWS`): row -> (tile, K-waves, split K, cache modifier), M <= 768. Use the
  fused chain only where the table below wins (rows <= 128); above it the separate chain stays. Calls share one
  split-K workspace per device: serial on one stream, CUDA-graph safe after `warmup_postmix_gemm`.

## Performance
One MI355X (shared), graph replay of 100 distinct inputs in cold weight copies, rounds A B B A, us per
sub-layer boundary, MXFP8 output (`benchmarks/gemm/mxfp8_postmix_hc4_n5120_k8192/benchmark_chain.py`). base = wo_b
decode GEMM to bf16 y (tuned tile, K-waves, split K, cache modifier), then the better of this repository's seam and AITER
`mega_mhc` (post-mix included). plain = `postmix_gemm` + seam without post-mix; gates = `postmix_gates_gemm` + finisher.

| rows | base | plain | gates | gates / base |
|---:|---:|---:|---:|---:|
| 16 | 20.22 | 18.96 | 15.57 | 0.77 |
| 32 | 22.58 | 22.13 | 18.57 | 0.82 |
| 64 | 25.77 | 26.06 | 23.01 | 0.89 |
| 128 | 32.77 | 34.07 | 32.36 | 0.99 |
| 192 | 37.93 | 41.16 | 37.60 | 0.99 |
| 256 | 41.87 | 45.68 | 43.53 | 1.04 |
| 384 | 52.98 | 56.27 | 53.56 | 1.01 |
| 512 | 69.64 | 81.13 | 78.06 | 1.12 |
| 768 | 94.11 | 101.94 | 98.68 | 1.05 |

The epilogue moves the same R, R' bytes (41 KB read + 41 KB written per row) as the separate seam, plus the records, so
the fusion wins only where the seam's dependent chain (6-9 us) dominates; the decode mainloop bounds the rest, and the
chain stays at two launches. Limit: at 16 rows the HBM weight stream, 7.91 us for 41.9 MB; the GEMM takes 11.6 us (68%).
At 192 / 384 rows HBM (weights re-read per m-block + epilogue) at 2.6 of 5.3 TB/s (49%); fp8 MFMA at 384 rows 0.65 of
4.06 PF (16%).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_postmix_hc4_n5120_k8192 -q`
- `python3 benchmarks/gemm/mxfp8_postmix_hc4_n5120_k8192/benchmark_chain.py`
