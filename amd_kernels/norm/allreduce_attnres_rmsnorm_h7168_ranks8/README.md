# Eight-rank all-reduce + AttnRes mix + RMSNorm, hidden 7168

## What it does

`AttnResComm.allreduce_attnres_rmsnorm` fuses what a Kimi-K3.1 decode layer runs after every TP8 row-parallel GEMM
(`o_proj`, the MoE final reduce, the dense MLP): the all-reduce of the bf16 partial, the AttnRes block mix of the running
prefix (`attn_res`) and the RMSNorm that follows, in one launch. Partial bf16 `[M, 7168]`, prefix bf16 `[M, 7168]`
updated in place, bank bf16 `[M, <=8, 7168]` (row `block_write_idx` written when a block ends), weights bf16 `[7168]`,
output bf16 `[M, 7168]`; M 1..256, one workgroup per row. `reference.py` is the contract: the eight partials are added
in rank order in fp32 and rounded once to bf16, the prefix add rounds to bf16, the mix and the norm are fp32 with one bf16
rounding of the output. The prefix out and the bank row are bit for bit the reference, the output is within one bf16 ulp
of the fp64 evaluation; deterministic, no atomic adds into outputs, no quantization (the consumers are bf16 GEMMs).
Optional forms: no prefix (a block's first source), no bank (a plain all-reduce + add + RMSNorm), no output norm,
`reduce=False` for an already reduced delta (no communication), `out_packed` for the tile-packed output layout of the
MFMA GEMMs. An all-gather can be expressed as the sum: partials that are -0.0 outside a rank's own column shard reduce
bit for bit to the gathered tensor (`test_replicated_layout.py`).

The exchange is a two-phase reduce-scatter / all-gather by posted writes into persistent uncached mailboxes with the data
as its own signal (module docstring of `kernel.py`): nothing is registered per graph, no flags, no host work, no staging.

## Entry points

- `AttnResComm`: the communicator, a collective on the eight TP ranks (CPU group needed); 14.7 MiB of mailboxes per rank.
- `AttnResComm.allreduce_attnres_rmsnorm` / `allreduce_attnres_rmsnorm`: the fused call -> (out, prefix out).
- `AttnResComm.close`: collective, after the last call completed.

One communicator per stream; every rank makes the same calls in the same order with the same row counts. No row-count
dispatch (one workgroup per row); the variants (prefix, output norm, bank, reduce, packed output) are the entry's
warm-up modes (`amd_kernels.warm_up`), compiled in `host.py`.

## Performance

MI355X, 8 ranks, graphs of 27-100 distinct input sets (larger than the LLC), min of 15 rounds, MAX over ranks, us per
call; the call is a pre-MLP one (prefix + delta, bank of 4 rows, output norm). Baseline: AITER custom all-reduce, then
vLLM's Triton AttnRes (regtile) launch. Chain floor = 1.57 us empty-kernel period + 0.5 load + 2 x (0.77 flight + 0.23
read) + 0.1 reduce + 0.6 tail + both phases' bytes at the measured 355 GB/s xGMI all-to-all (`benchmark_allreduce.py`
prints it with the HBM, xGMI and FLOP rooflines: 3-20% of 6.4 TB/s, 9-53% of 355 GB/s, under 2% of fp32 peak).

| M | stock | all-reduce alone | fused | speedup | chain floor (% of fused) |
|---:|---:|---:|---:|---:|---:|
| 12 | 11.90 | 5.91 | 9.88 | 1.20 | 5.62 (57) |
| 24 | 13.11 | 6.93 | 10.59 | 1.24 | 6.47 (61) |
| 40 | 14.98 | 8.19 | 11.82 | 1.27 | 7.60 (64) |
| 65 | 17.56 | 10.43 | 13.50 | 1.30 | 9.36 (69) |
| 80 | 19.10 | 11.10 | 14.94 | 1.28 | 10.42 (70) |
| 128 | 23.08 | 14.38 | 19.63 | 1.18 | 13.82 (70) |
| 192 | 29.30 | 19.55 | 25.39 | 1.15 | 18.34 (72) |

The limiter is the dependent chain, not a bandwidth or compute roof: two phases of write -> flight -> poll around a
one-wave reduce, ranks that start up to 1.3 us apart (the same skew the stock all-reduce pays), the bank statistics on
the one CU of a row (VALU issue, 0.2-0.4 us per bank row at M=12) and a tail.

## Tests

```bash
python3 -m pytest tests/norm/allreduce_attnres_rmsnorm_h7168_ranks8 -q                 # 8 GPUs
python3 benchmarks/norm/allreduce_attnres_rmsnorm_h7168_ranks8/benchmark_allreduce.py   # 8 GPUs
python3 benchmarks/norm/allreduce_attnres_rmsnorm_h7168_ranks8/ceilings.py --mode ping  # 8 GPUs
```
