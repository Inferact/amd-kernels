# Vocabulary-sharded embedding lookup: the shape-agnostic entry

## What it does
One factory for every vocabulary gather package: reads the hidden size from this rank's bf16 shard [V / world, H]
and the world size from the group, and builds that shape's package's `VocabGather`, which maps every rank's shard
once (collective, no weight copy) and then looks token ids up by peer reads, bit for bit the package's reference
(the checkpoint rows; subnormals and ids outside the vocabulary as positive zero). Keys (H, world): (7168, 8)
(`vocab_gather_h7168_v163840_ranks8`), (6144, 2) and (6144, 4) (`vocab_gather_h6144`); the vocabulary is a runtime
argument. The caller calls the returned object (`gather(ids)`, `warm_up()`, `close()`): no wrapper sits between it
and the package. A shard or group no package serves gets a one-line reason before any collective, and the caller
keeps its own embedding; every rank must get the same answer (the factory is collective).

## Entry points
- `vocab_gather_common.vocab_gather_comm`: builds the gather (every rank, same order, after the weights load,
  before graph capture).
- `vocab_gather_common.vocab_gather_comm_blocker`: None when `vocab_gather_comm` serves this shard, rank and group,
  else why not (host checks only, no collective).
- `vocab_gather_common.prepare_vocab_gather_comm`: the gather, or the reason it cannot be built.
- `vocab_gather_common.vocab_gather_blocker`: None when a built gather serves these ids (1-D contiguous
  int32 / int64 on the shard's device), else why not; the caller asks once per call.
- `vocab_gather_common.vocab_gather_shapes`: the covered (H, world) keys, for a caller deciding at load time.
- `vocab_gather_common.available`: gfx950 with FlyDSL importable.

Dispatch: `host._ROUTES` maps (H, world) to the package, imported on first use.

## Performance
The entry adds nothing per lookup: the gather is the package's object. 8 MI355X, hidden 7168, 20480 rows a shard,
random ids, min of 20 rounds, the maximum over ranks; graph replay of 20 lookups on their own ids; host us per eager
call from a 100-call enqueue loop; reason = `vocab_gather_blocker`.

| tokens | replay direct us | replay entry us | host direct us | host entry us | added us | reason us |
|---|---|---|---|---|---|---|
| 1 | 2.56 | 2.59 | 10.68 | 10.65 | -0.03 | 0.76 |
| 128 | 9.43 | 9.42 | 10.81 | 10.69 | -0.12 | 0.75 |
| 2048 | 121.72 | 123.64 | 10.93 | 10.70 | -0.23 | 0.77 |

## Tests
```bash
python3 -m pytest tests/embedding/vocab_gather_common -q   # test_entry_ranks8.py needs 8 GPUs
python3 benchmarks/embedding/vocab_gather_common/benchmark_routing.py   # 8 GPUs
```
