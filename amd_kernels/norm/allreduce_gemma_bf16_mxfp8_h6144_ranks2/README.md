# Two-rank all-reduce + Gemma norm, bf16-row MXFP8, hidden 6144

## What it does

`Bf16ArNormComm` fuses the TP2 all-reduce of a bf16 partial with the residual add, Gemma RMSNorm and (optionally) the
MXFP8 quant, in one launch, for the MiniMax-M3.1-Flash DSpark draft on TP2 (the draft's o_proj / down stay TP
partials; nine quantized tails and one final bf16 tail per draft step). Input bf16 `[rows, 6144]` contiguous, rows
1-2048 (the draft runs 7 per request); residual bf16, unit column stride, row stride divisible by 8; weight bf16
`[6144]` contiguous. Codes row-major or (16, 16)-tiled, padded to `code_tile_rows`; scales in AITER's shuffled layout,
padded to 32 rows; padding rows are not written. Contract (`reference.py`): `bf16(ftz(ftz(p0) + ftz(p1)))` at every
row (commutative: AITER's one- and two-stage bits at two ranks); the residual add is fp32; the norm (weight + 1) uses
the 12-wave square tree of the plain norm; the quant reads the bf16-rounded norm (the draft's rule). Deterministic;
atomics only on flags.

## Entry points

- `Bf16ArNormComm`: the communicator, a collective on two ranks with the device current; ~50 MiB shared per rank.
- `Bf16ArNormComm.warm_up`: collective; compiles every row tier, both code layouts, bf16 out, both residual forms.
- `Bf16ArNormComm.ar_add_rmsnorm_mxfp8`: all-reduce + add + norm + quant -> (codes, scales, residual out).
- `Bf16ArNormComm.ar_add_rmsnorm`: the same with bf16 out -> (out, residual out).
- The owned-residual form of both (from `host.OWNER_MIN_ROWS`): each rank reduces, norms and quantizes its half of
  the rows and stages the result (codes + scales, or the bf16 row) for the peer, so the residual is read and written on
  `owned_rows(rows)` only. Pass a full residual or the residual out of an owned call with the same row count. Outputs
  equal the all-row form bit for bit.
- `Bf16ArNormComm.capture`: a no-op context matching the four-rank API (eager and graph calls run the same kernel).
- `Bf16ArNormComm.close`: collective, after graph use has ended.

One communicator per stream, the same calls in the same order on every rank. `host.config` picks rows per batch and
the role grid by row count.

## Performance

MI355X TP2 (one xGMI link, ~56 GB/s per direction). Graphs over distinct inputs beyond the 256 MB MALL, MAX over
ranks, median us. Baseline: AITER custom all-reduce + the one-GPU bf16 norm + the FlyDSL quantizer. MXFP8, tile 16.

| rows | 7 | 28 | 56 | 224 | 448 | 896 | 1792 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 11.17 | 16.35 | 23.39 | 63.16 | 114.95 | 218.35 | 422.80 |
| this entry, all-row | 6.86 | 11.39 | 17.71 | 57.15 | 107.71 | 211.52 | 415.82 |
| this entry, owned residual | (all-row) | 11.06 | 15.37 | 45.45 | 86.90 | 168.76 | 336.26 |

## Tests

```bash
python3 -m pytest tests/norm/allreduce_gemma_bf16_mxfp8_h6144_ranks2 -q                             # 2 GPUs
python3 benchmarks/norm/allreduce_gemma_bf16_mxfp8_h6144_ranks2/benchmark_allreduce_ranks2.py      # 2 GPUs
```
