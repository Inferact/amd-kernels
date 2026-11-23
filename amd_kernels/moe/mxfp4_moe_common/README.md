# MXFP4 MoE block entries (routed experts + MXFP8 shared expert), routed by shape

## What it does

A whole MoE block on one GPU's TP shard: sigmoid-bias router + top-k, MXFP4 routed experts (SwiGLU-OAI), MXFP8
shared expert, reduced to bf16 [M, H] or returned as partials (slots, shared) for a fused all-reduce. The weight build
reads (hidden, intermediate per GPU, experts) from the checkpoint-layout w13 [E, 2I, H / 2] / w2 [E, H, I / 2] and
takes top_k: (6144, 768, 128, 4) -> `mxfp4_h6144_i768_e128_top4`, (6144, 1536, 128, 4) ->
`mxfp4_h6144_i1536_e128_top4`. The op follows the package that built its weights. Numerics are the package's
`reference.py`.

## Entry points

- `mxfp4_moe_common.available`: gfx950 with FlyDSL importable.
- `mxfp4_moe_common.prepare_mxfp4_moe_weights`: the package's `MoeWeights` from the checkpoint tensors, or the
  reason (also `mxfp4_moe_weights`, `mxfp4_moe_weights_blocker`); `warm_up_mxfp4_moe` after the caller frees the
  checkpoint copies.
- `mxfp4_moe_common.mxfp4_moe`: the op; `mxfp4_moe_blocker`, `mxfp4_moe_shapes`.
- `mxfp4_moe_common.mxfp4_moe_set_dbo_query`: the DBO-microbatch query of every routed package.

Dispatch: `_ROUTES`, `_WEIGHTS` in `moe.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/moe/mxfp4_moe_common -q   # host only
```
