# Fused Markov draft chain: the shape-agnostic entry

## What it does
One entry for every fused Markov draft chain package: reads (vocabulary, Markov rank) from `w1` [V, R] and calls that
shape's package, which runs all k draft steps (bias GEMV, logits into the draft logits cache, Gumbel draw, argmax into
`draft_tokens`) in one persistent launch, bit for bit the package's contract. Shape: (163840, 512)
(`markov_gumbel_fused_v163840_r512`: 1-48 requests, 1-8 steps, bf16 weights, base logits and cache; every operand on the
current device). The launch's 256 workgroups must be co-resident: nothing else may hold CUs of the device while it runs.
A call no package serves gets a one-line reason and the caller runs its own loop.

## Entry points
- `markov_gumbel_common.markov_gumbel_fused`: the chain; the routed package's function, which warms itself up on its
  first call of a step count.
- `markov_gumbel_common.markov_gumbel_fused_unsupported_reason`: None when `markov_gumbel_fused` serves these
  arguments, else why not (host checks only, safe inside graph capture); the caller asks once per call.
- `markov_gumbel_common.markov_max_rows`, `markov_gumbel_common.markov_scratch_dwords`: the row limit and the int32
  scratch size of `w1`'s package, to size the scratch (zeroed once, then owned by the kernel) at load time.
- `markov_gumbel_common.markov_gumbel_fused_shapes`: the covered (V, R) keys, for a caller deciding at load time.
- `markov_gumbel_common.available`: gfx950 with FlyDSL importable.

Dispatch: `host._SHAPE_HOSTS` maps (V, R) to the package, imported on its first call.

## Performance
The entry adds the dispatch only: a graph holds the package's kernel, the eager call one dict lookup. One MI355X,
min of 20 rounds; graph replay of 8 chains on their own base logits, T 1.0; host us per eager call from a 20-call
enqueue loop; reason = `markov_gumbel_fused_unsupported_reason`.

| rows, steps | replay direct us | replay entry us | host direct us | host entry us | added us | reason us |
|---|---|---|---|---|---|---|
| 8, 4 | 103.45 | 103.25 | 12.10 | 12.05 | -0.05 | 5.81 |
| 20, 3 | 89.89 | 90.00 | 12.75 | 12.79 | +0.04 | 5.78 |
| 48, 3 | 105.78 | 106.31 | 13.10 | 12.89 | -0.21 | 5.78 |

## Tests
```bash
python3 -m pytest tests/sampling/markov_gumbel_common -q
python3 benchmarks/sampling/markov_gumbel_common/benchmark_routing.py
```
