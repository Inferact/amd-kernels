# mHC seam, finisher and tail entries, routed by shape

## What it does

The mHC seam (post-mix, gates, collapse, RMSNorm, optionally MXFP8 and the streams' mean), its finisher from the post-
mix GEMM's tile records, and the step-end tail (post-mix, collapse, final RMSNorm), each keyed by (residual streams,
hidden): (4, 5120). `mhc_post_pre_rmsnorm_blocker` also declines the rows where AITER's MXFP8 mega-mHC seam was faster
(the seam wins at 1-288 and 513-720 rows; the `aux_mean` variant at any row count). Numerics are the packages'
`reference.py`.

## Entry points

- `mhc_post_pre_rmsnorm`: the seam; `mhc_post_pre_rmsnorm_blocker`, `_shapes`, `mhc_pack_gate_weight`,
  `warmup_mhc_post_pre_rmsnorm`.
- `mhc_gate_finish`: the finisher; `mhc_gate_finish_blocker`, `mhc_gate_finish_shapes`.
- `mhc_post_collapse_rmsnorm`: the tail; `mhc_post_collapse_rmsnorm_blocker`, `_shapes`,
  `warmup_mhc_post_collapse_rmsnorm`.

Dispatch: `_SEAM`, `_FINISH`, `_TAIL` and the row ranges `_SEAM_ROWS` in `host.py`; a shape package is imported on the
first call of its key. Callers import the names from the category (`amd_kernels.<category>`).

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries.py -q   # host only
HIP_VISIBLE_DEVICES= python3 -m pytest tests/test_op_entries_prepare.py -q   # host only
```
