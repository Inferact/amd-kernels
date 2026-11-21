# bf16 GEMM, N 20480 x K 7168 (the TP8 lm_head vocabulary shard)

## What it does
`out [M, 20480] = x [M, 7168] @ W^T`, bf16 operands, fp32 sums, one bf16 rounding: the TP8 vocabulary shard of the
Kimi-K3.1 lm_head (`lm_head.weight` bf16 [163840, 7168], rank r holds rows [20480 r, 20480 (r + 1)); W stays
row-major, no load-time transform). One tensor serves the target lm_head and the DSpark draft head (target-tied).
Contract (`reference.py`): the fp64 product rounded once to bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3,
deterministic bits, a row's bits independent of the batch size and the row's position within one kernel band; NaN /
inf propagate as in an fp32 sum. No atomics on the output. Up to 96 rows the band-stream kernel (`band.py`, adapted
from AITER's FlyDSL decode GEMM): 256 workgroups, 80 weight rows and all activation rows each, fixed K order from a
staggered start. From 97 rows the split-K kernel (`splitk.py`): 128 pairs of workgroups on one XCD, each pair owning
160 columns and each member one half of K, so the activation traffic of a workgroup halves; the two partial tiles meet
through a fixed-order L2 exchange (workspace + self-clearing flags), member order 0, 1 for every row.

## Entry points
- `bf16_gemm_n20480_k7168`: the logits of 1-256 rows, graph-safe, no host sync; optionally fuses the hidden-row gather
  and pushes the rows straight to their owner rank (or every rank) of `collectives/all_to_all_bf16_n20480_ranks8`.
  An `amd_kernels.warm_up` entry: the first call of each mode (contiguous / gather / owner / all) compiles its bands.
  The split-K bands share one workspace per device: two split-K launches must not run concurrently on different streams.
- `supports`: whether a weight is the [20480, 7168] bf16 contiguous shard and there is no bias.

`host.band_of` gives the kernel configuration of a row count (the dispatch is `host._kernel`);
`RowAllToAll.gemm_exchange` of the collective picks the GEMM's own push or a local write plus one push by row count.

## Performance
One MI355X, graph replay of six calls over three weight copies (880 MB), minimum of 15 rounds of ten replays, us per
call. Stock = AITER's `tgemm.mm` as vLLM calls it (the FlyDSL band-stream kernel up to 128 rows, hipBLASLt above).
Ceiling = max(293.6 MB at 6.5 TB/s measured HBM read, 2 M N K at the 1.5 PF measured library bf16 GEMM). Stream-only
floor = a kernel that only streams the split-K operands through the same rings (no math, no exchange).

| M | 1 | 12 | 24 | 40 | 65 | 80 | 96 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 43.1 | 43.3 | 43.9 | 45.9 | 51.6 | 53.3 | 55.2 | 60.0 | 100.0 | 103.9 | 144.8 |
| this entry | 43.2 | 43.3 | 44.0 | 45.9 | 50.4 | 51.6 | 53.9 | 57.6 | 65.4 | 66.2 | 85.4 |
| ceiling | 45.2 | 45.3 | 45.4 | 45.5 | 45.7 | 45.8 | 46.0 | 46.3 | 46.5 | 46.8 | 50.1 |
| % of ceiling | 104 | 104 | 103 | 99 | 91 | 89 | 85 | 80 | 71 | 71 | 59 |
| stream-only floor of the split-K design | - | - | - | - | - | - | - | 47.3 | 48.8 | 50.2 | 61.2 |

Up to 40 rows the weight stream (6.95 TB/s alone) is the limit; above, the L2 -> LDS path carries the weights and the
activation re-reads together (~12 TB/s), so the activation traffic adds to the weight time: the band kernel at 80 rows
sits at the floor of its design (50.7-52.3 us), the split-K pairs cut the traffic of a workgroup from 3.9 MB to 2.5 MB
at 192 rows. The split-K kernel is 10-16 us above its floor at 128-192 rows: the K loop 1-7 us (MFMA and LDS fragment
reads next to the DMAs), the exchange and the output store 4-5 us, the slowest workgroup 2-4 us after the mean.

## Tests
- `python3 -m pytest tests/gemm/bf16_n20480_k7168 -q` (`test_push.py`: 8 GPUs)
- `python3 benchmarks/gemm/bf16_n20480_k7168/benchmark_gemm.py [M ...]`
