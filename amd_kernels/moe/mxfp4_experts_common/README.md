# MXFP8 x MXFP4 routed-expert entries, routed by shape

## What it does

One rank's routed experts (MXFP8 activations, MXFP4 weights) keyed by (hidden, per-rank intermediate, top_k): (5120,
2304, 6) and (5120, 2304, 3); and the fused expert-parallel MoE (dispatch, both GEMMs and combine in one launch per
rank, a collective) keyed by (ranks, hidden, intermediate, top_k): (8, 5120, 2304, 6) ->
`fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8`. Numerics are the packages' `reference.py`.

## Entry points

- `mxfp4_experts`: one rank's experts; `mxfp4_experts_blocker`, `mxfp4_experts_shapes`, `mxfp4_expert_weights`,
  `prepare_mxfp4_expert_weights`.
- `expert_parallel`: the fused EP object (one per process); `expert_parallel_blocker` (decide it the same on every
  rank), `expert_parallel_shapes`, `prepare_expert_parallel`, `fused_ep_expert_weights`.

Dispatch: `_ROUTES`, `_FUSED_EP` in `host.py`; a shape package is imported on the first call of its key. Callers
import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
