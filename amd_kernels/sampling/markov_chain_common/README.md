# Markov draft chain entry, routed by shape

## What it does

Every draft step of every request's Markov chain (Gumbel draws over the vocabulary) in one launch, keyed by (vocab,
Markov rank, draft steps): (129280, 256, 5) -> `markov_chain_gumbel_bf16_v129280_r256_s5`. The blocker also declines
more than 160 requests, where the sequential steps were faster. Numerics are the package's `reference.py`.

## Entry points

- `markov_chain`: the chain; `markov_chain_blocker`, `markov_chain_shapes`, `warmup_markov_chain`.

Dispatch: `_ROUTES` and the request ranges `_REQUESTS` in `host.py`; a shape package is imported on the first call of
its key. Callers import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
