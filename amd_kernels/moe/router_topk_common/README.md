# Sqrt-softplus top-k router entry, routed by shape

## What it does

The gate GEMM + biased sqrt-softplus top-k of each row, keyed by (hidden, experts, top_k): (5120, 384, 6) ->
`router_sqrtsoftplus_top6_h5120_e384`, (5120, 128, 3) -> `router_sqrtsoftplus_top3_h5120_e128`. Numerics are the
packages' `reference.py`.

## Entry points

- `router_topk`: the router; `router_topk_blocker`, `router_topk_shapes`, `router_shuffle_gate`, `warmup_router_topk`.
- `prepare_router_topk`: the router with its gate shuffled and every row bucket compiled (a handle), or the reason.

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
