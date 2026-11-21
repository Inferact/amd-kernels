# Latent MoE decode entries: router, routed experts, latent tail, layer tail

## What it does
One shape-agnostic entry per op of the latent MoE decode step; a caller imports this package, never a shape package.
An entry reads its key from the operands (the gate weight and top-k, the experts' `w13`, the communicator or runner),
imports the shape package that serves the key on its first call and passes the call through unchanged: numerics, CUDA
graph behaviour and warm-up are the shape package's. Communicators and runners are built before any activation
exists: their factories take the key's dims as keywords and read the world size from the group; their calls route by
the object. `<op>_unsupported_reason` takes the op's arguments and returns None or a one-line reason (no kernel for the
key, rows out of range, an operand layout the package rejects); it is host-only (no device read, no allocation, safe
inside graph capture), and the caller falls back on a reason. The op only looks the key up and raises `ValueError`
with the same text off the table. `<op>_shapes` lists the covered keys for load-time decisions.

## Entry points
- `available`: gfx950 with FlyDSL (every routed package's requirement).
- `router_sigmoid_bias_topk`: the router (top-k weights and ids; with `aux` also the experts' sort);
  `router_sigmoid_bias_topk_weight_reason`: the per-layer load-time check of the gate weight and bias.
- `router_aux`: the sort tensors of one row count that the router fills.
- `moe_weights`: views of a layer's routed expert weights; `moe_sort_from_aux`: the sort from `router_aux`'s tensors
  (its reason checks the router -> experts hand-off); `moe_sort`: the sort by AITER; `moe_sort_capacity`: its rows.
- `quantize_moe_activations`: the experts' MXFP4 activation codes and scales.
- `mxfp4_moe_experts`: the routed experts, summed or as the slot buffer.
- `latent_tail_comm`: the latent tail communicator; `allreduce_latent_rmsnorm_up_proj`,
  `allreduce_slots_rmsnorm_up_proj`: the tail on a summed partial or on the experts' slot buffer.
- `layer_tail_runner`: the layer tail's per-rank state; `layer_tail_launch`: one layer's tail from a build mapping
  and the operands; the runner supplies its own arguments. Keys the package's build lacks are ignored, but a build
  asking for a fused MLA / DSA part (`m1b_qlen`, `m1c`, `dsa`) gets a reason.

Dispatch: `_SHAPE_HOSTS` (key -> shape package module) and `_shape_host` in `router.py`, `experts.py`,
`latent_tail.py`, `layer_tail.py`.

## Performance
The entries add only the dispatch (a cached key lookup and the call); a graph holds the package's kernels alone.
MI355X, us per call, both calls with the same arguments; graph: 20 calls a graph, min of 15 rounds; host: Python time
of an eager call, min of 50 batches of 100; 8-rank ops: max over ranks (`benchmark_routing.py`).

| op | M | graph package | graph entry | host package | host entry | host added |
|---|---|---|---|---|---|---|
| router + sort | 1 | 7.74 | 7.72 | 13.71 | 14.42 | +0.71 |
| router + sort | 192 | 15.14 | 15.16 | 13.79 | 14.41 | +0.62 |
| experts | 1 | 10.78 | 10.83 | 25.85 | 27.35 | +1.50 |
| experts | 192 | 126.49 | 126.41 | 25.32 | 25.87 | +0.54 |
| latent tail (8 ranks) | 1 | 10.04 | 10.02 | 12.42 | 12.62 | +0.20 |
| latent tail (8 ranks) | 256 | 23.31 | 23.31 | 12.34 | 12.78 | +0.45 |
| layer tail (8 ranks) | 4 | 63.61 | 63.57 | 17.84 | 20.18 | +2.34 |
| layer tail (8 ranks) | 16 | 124.11 | 124.09 | 18.00 | 20.12 | +2.13 |

The layer tail's host cost is forwarding its 20 keyword operands (the package call is timed with a prebuilt build).

## Tests
```bash
python3 -m pytest tests/moe/latent_moe_common -q   # test_routing: no GPU; test_*_ranks8: 8 GPUs
python3 benchmarks/moe/latent_moe_common/benchmark_routing.py
torchrun --nproc_per_node 8 benchmarks/moe/latent_moe_common/benchmark_routing.py
```
