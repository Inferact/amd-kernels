# RMSNorm + MXFP8 entries, routed by shape

## What it does

RMSNorm of bf16 rows quantized to MXFP8, keyed by hidden: 5120 -> `rmsnorm_mxfp8_h5120`; and the dual RMSNorm (q to
MXFP8, kv to bf16), keyed by (q dim, kv dim): (1280, 512) -> `rmsnorm_dual_mxfp8_q1280_kv512`. Numerics are the
packages' `reference.py`.

## Entry points

- `rmsnorm_mxfp8`: `rmsnorm_mxfp8_blocker`, `rmsnorm_mxfp8_shapes`, `warmup_rmsnorm_mxfp8`.
- `rmsnorm_dual_mxfp8`: `rmsnorm_dual_mxfp8_blocker`, `rmsnorm_dual_mxfp8_shapes`, `warmup_rmsnorm_dual_mxfp8`.

Dispatch: `_ROUTES`, `_DUAL` in `host.py`; a shape package is imported on the first call of its key. Callers import
the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
