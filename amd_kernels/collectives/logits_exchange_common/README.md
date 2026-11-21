# lm_head shard GEMM + logits exchange entry (vocabulary-sharded logits rows to their owners or to every rank)

## What it does

The shape-agnostic entry of the lm_head exchange: the factory reads the shape from keyword dims (vocabulary shard N,
hidden K) and the group (world size), builds the communicator of the shape package that serves it (imported on first
use) and takes the package's first-use buffers, so no call allocates (graph capture). The ops are the communicator's
methods, as in the package: `gemm_exchange` runs the [N, K] shard GEMM of the rows `x[row_indices]` and pushes its
output to the owners (`counts`, rows sorted by owner) or to every rank (`counts=None`); `recv` reads them.
Computation, dtypes and numerics are the package's (its `reference.py`): bf16, one rounding of an fp32 sum, an exact
byte exchange.

Routed: (20480, 7168, 8) -> `collectives/all_to_all_bf16_n20480_ranks8` (its GEMM: `gemm/bf16_n20480_k7168`). K is in
the key: `gemm_exchange` runs the GEMM of one weight shape. A call's reason routes by the communicator's type and
compares the weight with the communicator's key; it is host-only and takes the method's arguments.

## Entry points

- `available`: gfx950 with FlyDSL importable.
- `logits_gemm_exchange_shapes`: the covered (N, K, world), the load-time check of an lm_head shard.
- `logits_exchange_comm`, `logits_exchange_comm_unsupported_reason`: the collective communicator.
- `logits_gemm_exchange_unsupported_reason`: whether `comm.gemm_exchange(...)` with these arguments is served.

Dispatch: `host._SHAPE_HOSTS` ((N, K, world) -> host module of the shape package).

## Performance

The entry adds no call: the caller runs the communicator's methods, the same object the package builds. Eight MI355X,
all-gather mode, max over the ranks, min of 15 rounds, us per call: graph replay of 24 calls, host time of an eager
call (enqueue) and of the reason (`benchmark_routing.py`).

| rows | graph, package comm | graph, entry comm | host, package comm | host, entry comm | reason |
|---|---|---|---|---|---|
| 8 | 55.19 | 54.98 | 19.13 | 18.76 | 1.44 |
| 64 | 112.31 | 111.54 | 29.88 | 28.94 | 1.45 |
| 256 | 311.53 | 310.51 | 30.14 | 30.13 | 1.43 |

## Tests

- `python3 -m pytest tests/collectives/logits_exchange_common -q` (routing on CPU, `test_entry_ranks8.py` on 8 GPUs)
- `python3 benchmarks/collectives/logits_exchange_common/benchmark_routing.py` (8 GPUs)
