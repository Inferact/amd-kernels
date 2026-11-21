# Sharded drafter context projection: bf16 N-shard GEMM, eight-rank gather and RMSNorm, N 896 x K 35840

## What it does

The DSpark drafter's `context_proj` + `context_norm` of Kimi-K3.1 at TP8: the five aux hidden states [rows, 7168]
(K 35840) times the replicated bf16 weight [7168, 35840] (514 MB), then RMSNorm. Each rank keeps only its 896-row shard
(64 MB, packed at load), computes its 896 columns for all rows, and the eight shards are gathered and normalized on
every rank: bf16 [rows, 7168] out, 1-192 rows. Two launches: the split-K projection (fp32 slabs) and one kernel that
sums the slabs in a fixed order, posts the bf16 slice into every rank's mailbox, collects the eight slices and
normalizes (the mailbox protocol is in `reduce_gather.py`). Contract (`reference.py`): fp64 product rounded once to
bf16, then `bf16(x * rsqrt(mean(x^2) + eps) * w)` in fp32; cosine >= 0.99999, relative L2 <= 1e-3, bitwise repeatable
(no atomics into outputs).
The activations are read in the MFMA tile layout (`pack_aux` writes it where the drafter would
`torch.cat(aux, -1, out=staging)`); row-major reads cost +30% at 80 rows.

## Entry points

- `ProjComm`: the eight ranks' mailboxes and workspace; collective, one per stream, `close` on all ranks.
- `pack_weight`: this rank's 896 rows of the checkpoint weight, packed at load (free the row-major shard).
- `host.packed_rows_buffer`, `pack_aux`: the packed activation tiles and the aux-states-to-tiles copy of the rows.
- `linear_shard`, `pack_rows`: the shard GEMM alone (fp32 slabs) and a torch packing of row-major activations.
- `context_proj_gather_norm`: the normed [rows, 7168] on every rank (optionally into the caller's buffer), graph safe,
  same call order and row count on every rank; the first call of each tier warms up (`amd_kernels.warm_up`).

`host.TIERS` holds the tile / split / wave grid per row tier.

## Performance

Eight ranks, graph replay of 24 calls over six rotated 64 MB shards (replicated: two 514 MB copies), MAX over ranks,
min of 15 rounds, us per call. Floor = the 64 MB cold-read ceiling (11.3 us measured, launch included) + an 8-rank
all-gather of the output (5 us + 7/8 x bytes / 400 GB/s).

| rows | replicated GEMM + norm | sharded GEMM + AITER all-gather + norm | this entry | floor | % of floor |
|---|---|---|---|---|---|
| 40 | 91.7 | 30.9 | 22.3 | 17.6 | 79 |
| 65 | 95.1 | 34.4 | 26.4 | 18.3 | 69 |
| 80 | 97.5 | 35.8 | 27.9 | 18.8 | 67 |
| 192 | 143.9 | 45.4 | 47.0 | 22.3 | 47 |

`pack_aux` (eager, in place of the staging copy): 2.4 / 3.0 / 3.2 / 5.1 us.

## Tests

- `python3 -m pytest tests/gemm/bf16_linear_gather_n896_k35840_ranks8 -q` (test_context_proj, test_reduce_gather and
  test_warm_up: 8 GPUs; test_linear_shard: 1 GPU)
- `python3 benchmarks/gemm/bf16_linear_gather_n896_k35840_ranks8/benchmark_context_proj.py` (8 GPUs, needs vLLM)
