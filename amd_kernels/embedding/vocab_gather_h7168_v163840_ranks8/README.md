# Vocabulary embedding by owner reads on eight ranks, hidden 7168, vocabulary 163840

## What it does

Reads each token's embedding row straight from the rank that owns its vocabulary row: eight TP ranks keep their bf16
vocabulary shards (eight equal contiguous `[20480, 7168]` shards in rank order, no added-vocabulary gaps) and peers map
them once at load time over HIP IPC. One stateless kernel in place of the mask / gather / fill + all-reduce of
`VocabParallelEmbedding`: no weight copy, no payload sum, no staging buffer and no per-call handshake. Token `t` reads
shard `t // 20480`, row `t % 20480`; ids are a contiguous 1-D int32 or int64 tensor; the output is a contiguous bf16
`[rows, 7168]` on each rank. Caller: the Kimi-K3.1 target embedding (TP8). Contract (`reference.py`): finite rows bit
for bit; bf16 subnormals and signed zeros become +0 (AITER's fp32 flush when it adds the one owner to zeros); invalid
ids (< 0 or >= 163840) write +0; replays and eager calls are bitwise equal on every rank.

## Entry points

- `VocabGather`: the collective load-time mapping of the eight shards, on every rank once its shard is in its final
  allocation (publishes the GPU-written weight to peers); the weights stay alive and unchanged until `close()`.
- `VocabGather.__call__`: the `[rows, 7168]` bf16 rows of the ids, eager or in a CUDA graph; no per-call synchronization.
- `VocabGather.warm_up`: compiles the int32 and int64 variants (one per id dtype, any row count).
- `VocabGather.close`: collective on every rank, after all graph use.

No dispatch: one kernel per id dtype (`host.py`), rows a runtime argument.

## Performance

MI355X, eight ranks, graph replay of 20 distinct random token batches over a warm shard, minimum of nine rounds, MAX over
ranks, us per call. Baseline = vLLM's VocabParallelEmbedding in the same graph: compiled mask, `F.embedding`,
`masked_fill_` and AITER's custom all-reduce. Every rank embeds the same ids, so the link of the owner that holds the most
rows of the batch (`max rows`; ~30 rows at M 192 where the mean is 24) is the longest of the eight. Ceilings: the balanced
floor = launch gap (1.6 us) + one remote read round trip (1.0 us) + 7/8 of M x 14336 B at the measured 365 GB/s per-GPU
ingress of an 8-GPU all-read (owners round robin, no imbalance); the model = this kernel on rows of its own shard (`local`)
+ 1.5 us remote latency + `max rows` x 14336 B at the 52 GB/s per link of an 8-GPU all-read (measured; one owner for all
rows: 58 GB/s on its single link).

| M | 1 | 12 | 24 | 40 | 65 | 80 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| vLLM | 13.2 | 18.9 | 18.0 | 18.7 | 22.4 | 26.5 | 28.2 | 35.3 | 42.8 |
| this entry, random ids | 2.3 | 2.9 | 3.5 | 4.2 | 5.6 | 6.1 | 9.4 | 13.1 | 16.8 |
| % of balanced floor | 117 | 105 | 99 | 95 | 87 | 87 | 75 | 70 | 68 |
| model (busiest owner) | 3.7 | 4.4 | 5.1 | 5.9 | 7.1 | 7.9 | 10.3 | 13.2 | 15.5 |
| this entry, balanced owners | 2.3 | 2.6 | 2.9 | 3.5 | 4.6 | 5.1 | 7.0 | 10.4 | 14.5 |
| % of balanced floor | 113 | 116 | 119 | 115 | 106 | 105 | 100 | 88 | 78 |
| local rows only (launch + id chain + copy) | 2.0 | 2.0 | 2.0 | 2.1 | 2.2 | 2.4 | 2.8 | 3.1 | 3.2 |

Up to 128 rows the kernel is at or under both floors; at 192 rows with random ids it equals the busiest-link model (13.1
vs 13.2 us: the busiest link moves ~31 rows in 8.5 us = 52.5 GB/s, the measured per-link rate of the all-to-all), and with
balanced owners it is at 88 % of the balanced floor. The imbalance is in the ids, not the kernel: every row exists on exactly one rank, and
relaying rows of the busiest owner through another rank would save ~1.5 us of transfer for a second hop of ~2 us. Threads per
row (128 / 224 / 448 / 896, 7 / 4 / 2 / 1 workgroups per row) were swept at M 12 / 80 / 192: within 6 %.

## Tests

```bash
python3 -m pytest tests/embedding/vocab_gather_h7168_v163840_ranks8 -q          # 8 GPUs
python3 benchmarks/embedding/vocab_gather_h7168_v163840_ranks8/benchmark_gather.py [--ids balanced]   # 8 GPUs
```
