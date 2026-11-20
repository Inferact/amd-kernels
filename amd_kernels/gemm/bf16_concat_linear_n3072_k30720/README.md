# bf16 concat linear, N 3072 (one rank's shard) x K 30720, + the 2-rank exchange to N 6144

## What it does
The MiniMax-M3.1 DSpark draft's input fc at TP2: five bf16 taps [rows, 6144] (target layers 3, 17, 31, 45, 59; K
30720 = their concatenation) times `dspark.fc.weight` bf16 [6144, 30720], bf16 [rows, 6144] out on both ranks. Each
rank keeps its 3072-row weight slice (packed at load), computes its 3072 columns from the taps read in place (no
`torch.cat`), and the communicator exchanges the two shards over the pair's xGMI link into the caller's buffer. Every
output is one fp32 K sum (split-K slabs summed in a fixed order) rounded once to bf16; no quantization. Taps: unit
column stride, row stride % 8 == 0. Contract (`reference.py`): the fp64 product rounded once to bf16; per-row cosine
>= 0.99999, rel. L2 <= 1e-3, bitwise repeats. The slab count sets the fp32 summation order, so bits differ between
row tiers; within a tier they are deterministic. Caller: the TP2 dense decoder block (the draft's split input fc).

## Entry points
- `pack_weight`: the rank's 3072-row slice as a lossless permutation, at load; free the row-major slice.
- `bf16_concat_linear`: the local bf16 [rows, 3072] shard of 1-2048 rows from the taps in place, graph-safe.
- `warmup_concat_linear`: compiles the row tiers up to a row count.
- `comm.ConcatLinearComm`: the two-rank fc into the caller's contiguous bf16 [rows, 6144] buffer (IPC staging, up to
  2048 rows, the slab reduce fused into the exchange); eager only (its Python slot rotation cannot replay in a graph);
  both ranks make the same calls in the same order; its `warm_up` before serving, its `close` on both ranks.

`host.TIERS` picks the GEMM (kernel, tile, K slabs) by row count; `comm.EXCHANGE` the exchange group shape.

## Performance
Two ranks, eager fc as the draft calls it, cold rotated weights, MAX over ranks, median us (`benchmark_comm.py`).
Baseline = what vLLM runs at TP2: `torch.cat` + `F.linear` (hipBLASLt) on the replicated [6144, 30720] weight.
Serial roofline = max(weight + taps at 6.5 TB/s, bf16 at 2.5 PF) + the shard exchange at 64 GB/s.

| rows | 8 | 32 | 64 | 128 | 192 | 256 | 320 | 448 | 512 | 768 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| replicated F.linear | 67.7 | 69.8 | 77.3 | 96.8 | 129.4 | 126.0 | 162.7 | 179.8 | 225.2 | 275.4 | 343.0 | 628.8 |
| ConcatLinearComm | 35.7 | 41.2 | 46.7 | 62.4 | 81.7 | 98.1 | 116.5 | 147.1 | 166.0 | 249.3 | 308.7 | 601.0 |
| serial roofline | 29.9 | 32.4 | 35.8 | 42.5 | 49.3 | 56.0 | 62.8 | 76.8 | 87.8 | 131.7 | 175.6 | 351.2 |

## Tests
- `python3 -m pytest tests/gemm/bf16_concat_linear_n3072_k30720 -q` (exchange / compile: 2 GPUs)
- `python3 benchmarks/gemm/bf16_concat_linear_n3072_k30720/benchmark_concat_linear.py [--sweep]` (1 GPU, local)
- `python3 benchmarks/gemm/bf16_concat_linear_n3072_k30720/benchmark_comm.py [--sweep]` (2 GPUs)
