# Vocabulary embedding by owner reads on two or four ranks, hidden 6144

## What it does

Reads each token's embedding row straight from the rank that owns its vocabulary row: two or four TP ranks keep their
bf16 vocabulary shards (equal contiguous `[V, 6144]` shards in rank order, a plain contiguous partition with no
added-vocabulary gaps) and peers map them once at load time over HIP IPC. One stateless kernel in place of the
mask / gather / fill + all-reduce of `VocabParallelEmbedding`: no weight copy, no payload sum, no staging buffer and
no per-call handshake. Token `t` reads shard `t // V`, row `t % V`; ids are a contiguous 1-D int32 or int64 tensor;
the output is a contiguous bf16 `[rows, 6144]` on each rank. Caller: the MiniMax-M3.1 DSpark draft's embedding at
TP4 (V = 50016) through `decoder/dense_mxfp8_h6144_i3072_ranks4` and at TP2 (V = 100032) through
`decoder/dense_mxfp8_h6144_i6144_ranks2`; `embedding/vocab_gather_gemma_mxfp8_h6144` reuses the mapping. Contract
(`reference.py`): finite rows bit for bit; bf16 subnormals and signed zeros become +0 (AITER's fp32 flush when it adds
the one owner to zeros); invalid ids (< 0 or >= ranks x V) write +0; replays and eager calls are bitwise equal on every
rank.

## Entry points

- `VocabGather`: the collective load-time mapping of the shards (world 2 or 4), built on every rank after selecting
  its GPU and after this rank's shard reaches its final allocation; it checks equal shard sizes, publishes the
  GPU-written weight to peers and exchanges the IPC handles. The weights stay alive and unchanged until `close()`;
  never move or overwrite them while a captured graph may read them.
- `VocabGather.__call__`: the `[rows, 6144]` bf16 rows of the ids, eager or inside a CUDA graph; no inter-rank
  synchronization per call.
- `VocabGather.warm_up`: compiles the int32 and int64 variants (one per id dtype and world size, any row count).
- `VocabGather.close`: collective on every rank, after all graph use.

No dispatch: one kernel per id dtype and world size (`host.py`), rows a runtime argument.

## Performance

MI355X, real shard size (200064 / ranks rows); graph replay of 100 distinct batches (7 rows per request: one random
anchor + six mask ids), warm weights, max over ranks, median us per call. Stock = `VocabParallelEmbedding`'s mask /
gather / fill + AITER custom all-reduce (`benchmark_stock.py`). TP2 pulls half the anchors over one xGMI link.

| Requests (rows) | 1 (7) | 10 (70) | 16 (112) | 32 (224) | 64 (448) | 128 (896) | 256 (1792) |
|---|---|---|---|---|---|---|---|
| 2 ranks, stock | 24.81 | 37.99 | 48.47 | 75.79 | 128.46 | 235.09 | 438.18 |
| 2 ranks, `lookup` | 3.23 | 4.60 | 5.37 | 7.29 | 12.06 | 20.46 | 42.69 |
| 4 ranks, `lookup` | 3.21 | 4.23 | 4.72 | 6.02 | 8.90 | 15.51 | 28.46 |

## Tests

```bash
python3 -m pytest tests/embedding/vocab_gather_h6144 -q  # 4 GPUs and 2 GPUs
python3 benchmarks/embedding/vocab_gather_h6144/benchmark_gather.py --world 2  # or 4
python3 benchmarks/embedding/vocab_gather_h6144/benchmark_stock.py --world 2
```
