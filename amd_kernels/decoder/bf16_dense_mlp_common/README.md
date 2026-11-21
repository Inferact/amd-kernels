# Dense bf16 MLP shard entry (gate_up + SwiGLU, down on packed weights)

## What it does

The shape-agnostic entry of one rank's dense MLP shard: `out [rows, hidden] = down(silu(gate) * up)` of the packed
activations, the caller all-reduces it. It reads (hidden, inter) from the down weight [hidden, inter], calls the shape
package that serves it (imported on first use) and, for a call no package covers, says why in a one-line reason so
the caller falls back. Computation, dtypes and numerics are the routed package's (its `reference.py`); the entry
passes the arguments through, so its outputs are the package's bits.

Routed: (7168, 1792) -> `decoder/bf16_dense_mlp_h7168_i1792`. The workspace is built for one key; the reason checks it
against the call's weights. The packed weights keep the raw weights' shapes: pass the entry's `pack` output. The
reason is host-only (shapes, dtypes, strides, ints) and takes exactly the op's arguments.

## Entry points

- `available`: gfx950 with FlyDSL importable.
- `dense_mlp_workspace`, `dense_mlp_workspace_unsupported_reason`: the per-stream workspace of a (hidden, inter).
- `dense_mlp_pack_gate_up`, `dense_mlp_pack_down`: the load-time weight layouts.
- `dense_mlp_packed_tiles`, `dense_mlp_rows_buffer`: the packed activations of a row count.
- `bf16_dense_mlp`, `bf16_dense_mlp_unsupported_reason`, `bf16_dense_mlp_shapes`: the op, whether a call is served,
  the covered (hidden, inter).

Dispatch: `host._SHAPE_HOSTS` ((hidden, inter) -> host module of the shape package).

## Performance

The entry adds only the dispatch (one cached lookup per call); a CUDA graph holds the same kernels. One MI355X, min of
15 rounds, us per call: graph replay of 24 calls, host time of an eager call (enqueue), the routing alone (dispatch)
and the reason (`benchmark_routing.py`).

| rows | graph direct | graph entry | host direct | host entry | dispatch | reason |
|---|---|---|---|---|---|---|
| 16 | 15.36 | 15.32 | 16.64 | 15.38 | 0.14 | 2.00 |
| 60 | 23.65 | 23.72 | 16.39 | 16.06 | 0.15 | 1.98 |
| 144 | 34.57 | 34.65 | 16.39 | 16.20 | 0.15 | 2.06 |

## Tests

- `python3 -m pytest tests/decoder/bf16_dense_mlp_common -q` (routing on CPU, the rest on 1 GPU)
- `python3 benchmarks/decoder/bf16_dense_mlp_common/benchmark_routing.py` (1 GPU)
