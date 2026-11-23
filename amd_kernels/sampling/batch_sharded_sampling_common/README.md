# Batch-sharded sampling entries on a TP group, routed by shape

## What it does

Batch-sharded sampling with a vocabulary-sharded lm_head: the target's verify-logits exchange, the sampler-output
gather and the Gumbel Markov draft chain, held by one communicator built collectively. The key (vocab, world) is the
model's vocabulary and the group's size: (200064, 4) -> `batch_sharded_markov_v200064_h6144_r256_ranks4`, (200064, 2)
-> `..._ranks2`. The ops are the communicator's methods (`target_logits`, `gather_sampler_output`, `setup_markov`,
`markov_draft`). Draws and numerics are the package's `reference.py` (same seed, position, vocab id).

## Entry points

- `batch_sharded_sampling_common.available`: gfx950 with FlyDSL importable.
- `batch_sharded_sampling_common.batch_sharded_sampling_config_blocker`: load-time check of the batch shape.
- `batch_sharded_sampling_common.prepare_batch_sharded_sampling`: the communicator or the reason (also
  `batch_sharded_sampling`, `batch_sharded_sampling_blocker`, `batch_sharded_sampling_shapes`).
- `batch_sharded_sampling_common.batch_sharded_markov_blocker`: whether `setup_markov` serves a draft.

Dispatch: `_ROUTES` in `host.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/sampling/batch_sharded_sampling_common -q   # host only
```
