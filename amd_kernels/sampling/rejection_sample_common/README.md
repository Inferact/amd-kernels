# Standard rejection sampling entry, routed by shape

## What it does

Standard rejection sampling of the draft tokens against the target logits, keyed by (vocab, draft tokens): (129280, 5)
-> `rejection_standard_v129280_s5`. Numerics are the package's `reference.py`.

## Entry points

- `rejection_sample`: `rejection_sample_blocker`, `rejection_sample_shapes`, `warmup_rejection_sample`.

Dispatch: `_ROUTES` in `host.py`; a shape package is imported on the first call of its key. Callers import the names
from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
