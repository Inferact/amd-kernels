# Sequence-parallel decoder-layer hand-off entries on a TP group, routed by shape

## What it does

The hand-offs of a sequence-parallel decoder layer (each rank keeps a shard of the residual stream): the attention
output's MXFP8 pack + all-to-all + o_proj + post-attention norm + gather, the FFN output's reduce-scatter + next norm
+ gather; on pull collectives, or SDMA pushes per DBO microbatch (`sdma`). The key (hidden, attention output columns
per rank, world) is the layer's sizes and the group's: (6144, 2048, 4) -> `seq_parallel_handoffs_h6144_q2048_ranks4`,
(6144, 4096, 2) -> `seq_parallel_handoffs_h6144_q4096_ranks2`. The phases are the communicator's methods. Numerics
are the package's `reference.py`.

## Entry points

- `seq_parallel_handoffs_common.available`: gfx950 with FlyDSL importable.
- `seq_parallel_handoffs_common.seq_parallel_handoffs_layer_blocker`: load-time check of a layer's sizes.
- `seq_parallel_handoffs_common.seq_parallel_min_tokens`: the token count from which the layer pays.
- `seq_parallel_handoffs_common.prepare_seq_parallel_handoffs`: the communicator or the reason (also
  `seq_parallel_handoffs`, `seq_parallel_handoffs_blocker`, `seq_parallel_handoffs_shapes`).
- `seq_parallel_handoffs_common.seq_parallel_shard`: a rank's rows of every token in the communicator's sharding.

Dispatch: `_ROUTES` in `host.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/collectives/seq_parallel_handoffs_common -q   # host only
```
