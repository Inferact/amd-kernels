# Top-p mask of fp32 logits rows: the shape-agnostic entry

## What it does
One entry for every top-p mask package: reads the vocabulary from `logits` [n, V] and calls that vocabulary's package,
which masks the rows in place with -inf outside each row's top-p set, bit for bit as vLLM's split top-p chain (the
package's contract and reference). Vocabularies: 163840 (`top_p_mask_f32_v163840`, 1-64 rows) and 200064
(`top_p_mask_f32_v200064`, any row count; > 64 rows take an optional fp32 `buffer`). fp32 logits with unit column
stride, fp32 `p` [n], all on the current device. A call no package serves gets a one-line reason and the caller runs its
own top-p.

## Entry points
- `top_p_mask_common.top_p_mask`: the mask; the routed package's function, which warms itself up on its first call.
- `top_p_mask_common.top_p_mask_blocker`: None when `top_p_mask` serves these arguments, else why not
  (host checks only, safe inside graph capture); the caller asks once per call and falls back on a reason.
- `top_p_mask_common.top_p_mask_shapes`: the covered (V,) keys, for a caller deciding at load time.
- `top_p_mask_common.available`: gfx950 with FlyDSL importable.

Dispatch: `host._ROUTES` (`amd_kernels.routes.Routes`) maps (V,) to the package, imported on its first call.

## Performance
The entry adds the dispatch only: a graph holds the package's kernels, the eager call one dict lookup. One MI355X,
min of 20 rounds; graph replay of 40 masks on their own rows (restored before every replay); host us per eager call
from a 100-call enqueue loop; reason = `top_p_mask_blocker`.

| vocab, rows | replay direct us | replay entry us | host direct us | host entry us | added us | reason us |
|---|---|---|---|---|---|---|
| 163840, 8 | 63.76 | 63.72 | 7.83 | 7.94 | +0.10 | 3.07 |
| 163840, 64 | 222.30 | 222.74 | 8.02 | 8.20 | +0.18 | 3.10 |
| 200064, 100 (buffer) | 249.43 | 249.31 | 9.17 | 9.26 | +0.09 | 4.07 |

## Tests
```bash
python3 -m pytest tests/sampling/top_p_mask_common -q
python3 benchmarks/sampling/top_p_mask_common/benchmark_routing.py
```
