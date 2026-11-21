# bf16 GEMM, K 5120, fp32 columns plus 32 bf16 columns, up to 768 rows

## What it does

`x [M, 5120] bf16` times `[wa; wb]^T`, with `wa [NA, 5120]` (NA = 1024 or 512: the compressor's fused kv | score
projection) returned in fp32 `[M, NA]` and `wb [32, 5120]` (the indexer's weights_proj) rounded to bf16 `[M, 32]`, in one
launch (NA = 0: the weights_proj alone). The weights are a load-time layout (`shuffle_weight`: `[column tile, K step,
lane, 8]`, a permutation of the concatenated rows) so that a wave's fragment load is 1 KB in a row. Contract:
`reference.py`: the exact real sum rounded to fp32 / bf16. The kernel multiplies on MFMA with fp32 accumulation: each
wave sums a K part in ascending order, the parts of a workgroup are added in wave order through LDS, and the
K range of a tile is also split over `ksp` workgroups (`host.TILES`, 1-10): each stores its fp32 partial tile with
agent-scope stores, takes a ticket (a per-tile counter, zeroed once and reset by the last arrival, allocated before
graph capture), and the last workgroup of the tile adds the `ksp` partials in K order and writes the outputs. An output
is therefore one fixed fp32 sum for a given row count (repeat runs, graph replays and a different arrival order are
bit-identical; vs the contract: relative 1e-6, cosine above 0.99999, bf16 within one ulp). No atomics touch an output;
the partial / ticket scratch (8 MB, 512 tickets) is one per device, so two launches of this entry must not run
concurrently on different streams. Workgroups are laid out by XCD (workgroup i runs on XCD i % 8): an XCD takes a
contiguous run of [column group][row group][K split], so its L2 reads each weight / activation tile once.

## Entry points

- `gemm.gemm`: compute both outputs for the caller's activation rows into the caller's tensors.
- `gemm.shuffle_weight`: the load-time layout of `[wa; wb]` (call once, before graph capture).
- `gemm.supports`: whether the shapes fit (gfx950, M up to 768).

An entry (`amd_kernels.warm_up`): the first call per column split compiles every M tile. The shape to tile dispatch
(tile of 16-column x 16-row blocks, K parts in flight, waves) is `host.TILES`, the best of a sweep per row count.

## Performance

One MI355X; one graph of 100 launches with their own activations and 8 cycled weight copies, A B B A, median of 5 x 10
replays, us per call (a second run within 0.1 us). The baseline is two launches: `torch.mm(x, wa^T, out_dtype=float32)`
and `F.linear(x, wb)`; NA = 0 is the second launch alone.

| Rows | NA 1024 | Baseline | NA 512 | Baseline | NA 0 | Baseline |
|---|---|---|---|---|---|---|
| 16 | 5.30 | 16.55 | 4.78 | 16.27 | 3.51 | 6.95 |
| 48 | 6.70 | 15.00 | 5.61 | 14.80 | 3.76 | 7.07 |
| 96 | 8.05 | 17.58 | 6.57 | 15.24 | 4.04 | 7.28 |
| 192 | 11.14 | 21.42 | 9.16 | 20.25 | 4.66 | 9.83 |
| 384 | 16.91 | 27.25 | 13.98 | 23.34 | 5.30 | 9.86 |
| 768 | 26.74 | 28.58 | 17.92 | 26.15 | 5.93 | 9.58 |

The kernel wins at every row count and column split (NA = 0: 1.6-2.1x). Limit: NA 0 is bound by launch + cold operand
read (1.59 us empty graph node, 1.8 / 2.5 / 3.4 us cold read at 16 / 192 / 768 rows) plus the ticket / partial chain;
achieved 51-57% of that floor. NA 1024 at 768 rows is load-latency bound: 9.7 TB/s through the CUs, 31% of the 31 TB/s
L2-fed re-read (MFMA 10.5% busy).

## Tests

```bash
python3 -m pytest tests/gemm/bf16_k5120_n1056_f32_n32 -q
python3 benchmarks/gemm/bf16_k5120_n1056_f32_n32/benchmark_gemm.py
```

`test_k_split_deterministic_and_graph_safe`: 30 repeats with poisoned outputs, graph replays over changing activations,
zero tickets afterwards, for the K-split configurations of every column split.
