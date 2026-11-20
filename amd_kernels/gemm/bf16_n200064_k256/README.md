# bf16 projection, N 200064 x K 256, packed selected rows

## What it does
`out [n, 200064] = x[rows] @ W^T` for 0-128 selected rows, bf16 in, fp32 accumulate, one bf16 rounding: the DSpark
Markov projection (W = `markov_w2` [200064, 256]) of only the owner rank's requests in the MiniMax-M3.1-Flash
batch-sharded Markov draft loop (up to 64 rows per rank at TP4, 128 at TP2). W stays row-major (no copy, no
transform). The activation is packed bit for bit into the MFMA layout [pad / 16, 8, 4, 16, 8] (= [m16, k32, k8, row16,
feature8]) viewed as bf16 [pad, 256], pad = 16 / 32 / 64 / 128: `pack_bf16_rows` packs the first step's rows of a
table (`markov_w1`), and `sampling/gumbel_combine_markov_embed_v200064_r256` writes the next steps' rows in this layout.
Padding rows never reach an output. Contract (`reference.py`): the fp64 product rounded once to bf16; per-row cosine
>= 0.99999, rel. L2 <= 1e-3, deterministic. The sum order moves a few bf16 elements against hipBLASLt (<= 0.01%, up
to 192 ulp); sampled tokens and accepted lengths stay equal over 4.19M request-steps per class. Callers:
`sampling/batch_sharded_markov_v200064_h6144_r256_ranks4` and `..._ranks2`.

## Entry points
- `padded_rows`: the packed row count (16, 32, 64 or 128) for a number of selected rows.
- `pack_bf16_rows`: packs the selected rows of a table into the caller's packed buffer; indices are not checked (no
  host sync), graph-safe.
- `bf16_gemm_packed_rows`: the projection of the packed rows into a contiguous bf16 output; zero rows launch nothing.
- `warmup_bf16_gemm_packed_rows`: compiles the five GEMMs and four packs; call it before graph capture.

`host.py` picks the kernel and the pad by row count. No output atomics.

## Performance
One MI355X. The whole seven-step Markov chain (projection, Gumbel partials, tail):
four input sets, three rotated `markov_w2` copies, twelve chains per graph, five alternated rounds, median µs.
Baseline = projecting every request with the fused row-major tail. Projection alone: 100 packed inputs, three rotated
weights. The 65-128-row tier (TP2) is timed as the projection alone: at 128 rows two 64-row calls take 51.3 µs, the
HBM roofline (6.5 TB/s) is 23.6 µs.

| Requests | Owned rows | Chain, baseline µs | Chain µs | Projection alone µs |
|---|---|---|---|---|
| 10 / 16 / 32 (TP4) | 3 / 4 / 8 | 209.8 / 216.8 / 277.8 | 199.3 / 201.7 / 210.4 | 17.07 / 17.17 / 17.09 |
| 64 / 128 / 256 (TP4) | 16 / 32 / 64 | 302.8 / 432.9 / 665.5 | 244.8 / 312.1 / 443.8 | 17.23 / 20.74 / 26.02 |
| 192 / 256 (TP2) | 96 / 128 | | | 33.66 / 35.01 |

## Tests
- `python3 -m pytest tests/gemm/bf16_n200064_k256 tests/sampling/gumbel_combine_markov_embed_v200064_r256 -q`
- `python3 benchmarks/gemm/bf16_n200064_k256/benchmark_gemm.py`
