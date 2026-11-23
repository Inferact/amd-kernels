# Clamped SiLU-mul + MXFP8 entry, routed by shape

## What it does

`silu(min(g, limit)) * clamp(u, -limit, limit)` of a bf16 gate_up row, quantized to MXFP8 (row-major codes and e8m0
scales). The key is the intermediate size (half the gate_up width); 2304 -> `clamp_silu_mul_mxfp8_i2304`. Numerics are
the package's `reference.py`.

## Entry points

- `clamp_silu_mul_mxfp8`: the activation + quant; `clamp_silu_mul_mxfp8_blocker`, `clamp_silu_mul_mxfp8_shapes`,
  `warmup_clamp_silu_mul_mxfp8`.

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
