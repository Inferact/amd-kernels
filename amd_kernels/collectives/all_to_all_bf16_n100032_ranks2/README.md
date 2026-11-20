# bf16 owner-row all-to-all, 100032 columns per source, two ranks

## What it does

`RowAllToAll` moves the vocabulary-sharded logits rows of MiniMax-M3.1-Flash TP2 (DSpark target and draft sampling)
to the rank that owns each request, straight into the final `[owned rows, 200064]` layout, in place of RCCL's
`all_to_all_single` + permute of vLLM's batch-sharded sampler. Input: bf16 `[rows, 100032]` contiguous, rows <=
`max_rows` <= 2048, sorted stably by owner. Contract (`reference.py`): rank r receives rows
`[sum(counts[:r]), sum(counts[:r + 1]))` of both sources, concatenated along the vocabulary, every bf16 bit kept (NaN
payloads too). No atomic output adds. All the peer bytes cross one xGMI link: 16 workgroups pull them, 4 copy the own
rows (fewer remote workgroups contend less on the link: 16 beat 32-192 at every size).

## Entry points

- `RowAllToAll`: a collective; two IPC slots of `max_rows x 100032 x 2` B per rank (819 MB at 2048 rows).
- `RowAllToAll.warmup`: a collective before capture; compiles every stage grid and the pull.
- `RowAllToAll.__call__`: stage + pull of owner-sorted rows into a preallocated `[>= own rows, 200064]` bf16 output;
  the counts are a CUDA int32 `[2]`, the same on every rank, summing to the rows (device values, not checked).
- `RowAllToAll.slot`: this rank's send slot (pool address, slot bytes, device counter, flag address) for a producer
  that writes and publishes it itself (`gemm/bf16_n100032_k6144`'s publish); `RowAllToAll.pull` then receives.
- `RowAllToAll.close`: a collective, after every graph that uses the communicator is retired.

Both ranks call every exchange in the same order, empty receivers included; one communicator per stream. Counts may
change between graph replays (device sequence, no host state). `host.py` picks the stage grid by rows.

## Performance

Two GPUs (one xGMI link) of one MI355X node. `benchmark_exchange.py`: graph replay, inputs rotating over 256 MB, five
alternating rounds, MAX over ranks, median us, dense owners, stage + pull; baseline RCCL `all_to_all_single` +
permute. Link rate = the larger owner's peer bytes / time; the link floor is about 55.5 GB/s.

| requests (8 rows each) | baseline | this entry | link GB/s | draft (7 rows): baseline | this entry |
|---:|---:|---:|---:|---:|---:|
| 1 | 35.46 | 35.43 | 45.2 | 32.25 | 31.30 |
| 4 | 89.90 | 68.70 | 46.6 | 80.15 | 60.30 |
| 10 | 191.70 | 158.09 | 50.6 | 170.32 | 139.35 |
| 16 | 285.56 | 247.86 | 51.7 | 252.09 | 217.76 |
| 32 | 538.99 | 473.16 | 54.1 | 473.09 | 414.67 |
| 52 | 856.63 | 762.86 | 54.5 | 752.62 | 668.75 |
| 128 | 2062.83 | 1854.25 | 55.2 | 1808.71 | 1622.06 |

## Tests

```bash
python3 -m pytest tests/collectives/all_to_all_bf16_n100032_ranks2 -q                # 2 GPUs
python3 benchmarks/collectives/all_to_all_bf16_n100032_ranks2/benchmark_exchange.py  # 2 GPUs
```
