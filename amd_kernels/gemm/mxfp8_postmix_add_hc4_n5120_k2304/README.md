# Shared-expert down MXFP8 GEMM with the routed add and the mHC post-mix epilogue, N 5120 x K 2304 (gfx950)

## What it does
The FFN half of `gemm/mxfp8_postmix_hc4_n5120_k8192` (same mainloop, K-reduce, split-K hand-off and epilogues; read that
README): the shared expert's down projection reads the MXFP8 codes [M, 2304] and row-major E8M0 bytes [M, 72] that the
gate_up swiglu epilogue wrote (`gemm/mxfp8_decode_prequant`), as they are, and its epilogue forms
`y = bf16(bf16(acc) + routed)` with the routed experts' bf16 output [M, 5120] (the MoE combine's tensor, read once; the
no separate `routed + shared` launch) and then the post-mix `R'[j] = bf16(p[j] * y + sum_h c[h][j] * R[h])` with
the previous seam's gates (fma order of the seam: the new residual equals GEMM, bf16 add, seam post-mix bit for bit).
`postmix_gates_gemm` additionally writes x1, the per-wave gate-projection records and optionally the mean of the streams for
`norm/mhc_gate_finish_rmsnorm_h5120`. K = 18 steps of 128, so K-waves x split K is 1, 3 or 9. `reference.py` is the
contract (cosine >= 0.99999 against torch).

## Entry points
- `postmix_gemm`: down GEMM + add + post-mix; returns the new residual bf16 [M, 4, 5120].
- `postmix_gates_gemm`: the same + x1, records (+ mean); feeds `mhc_gate_finish`.
- `pack_gate_weight`, `preshuffle_weight`, `warmup_postmix_gemm`; dispatch in `host._ROWS` / `host._GATE_ROWS` (M <= 768).
  Fused chain only where the table wins (rows <= 192); above it the separate chain stays. One split-K workspace per
  device: serial on one stream, CUDA-graph safe after the warm-up.

## Performance
One MI355X (shared), graph replay of 100 distinct inputs in cold weight copies, rounds A B B A, us per FFN boundary
(`benchmarks/gemm/mxfp8_postmix_add_hc4_n5120_k2304/benchmark_chain.py`). base = decode GEMM to bf16, `routed + shared`
(one elementwise launch), then the better of this repository's seam and AITER `mega_mhc`; plain = `postmix_gemm` + seam
without post-mix; gates = `postmix_gates_gemm` + finisher; MXFP8 output.

| rows | base | plain | gates | gates / base |
|---:|---:|---:|---:|---:|
| 16 | 15.78 | 13.10 | 9.92 | 0.63 |
| 32 | 16.74 | 14.18 | 11.10 | 0.66 |
| 64 | 18.46 | 16.12 | 14.16 | 0.77 |
| 128 | 20.63 | 20.71 | 17.97 | 0.87 |
| 192 | 24.41 | 23.98 | 22.03 | 0.90 |
| 256 | 25.73 | 27.88 | 26.95 | 1.05 |
| 384 | 32.76 | 35.43 | 34.55 | 1.05 |
| 512 | 41.37 | 48.26 | 46.91 | 1.13 |
| 768 | 56.72 | 62.65 | 64.28 | 1.13 |

The mainloop is 4-22 us here, so the post-mix epilogue (the same 41 KB read + 41 KB written per row as the seam) is the
bulk of the chain above 256 rows and loses to the seam by the finisher's records; below 192 rows the removed launches
and the seam's dependent chain make it win. Limit: at 16 rows the graph node floor + weight stream (3.46 us for 11.8 MB,
2.6 us empty node); the GEMM takes 5.6 us (62%). At 384 rows the epilogue's HBM bytes (R read, R' write, routed read)
move at 2.8 of 5.3 TB/s (53%).

## Tests
- `python3 -m pytest tests/gemm/mxfp8_postmix_add_hc4_n5120_k2304 -q`
- `python3 benchmarks/gemm/mxfp8_postmix_add_hc4_n5120_k2304/benchmark_chain.py`
