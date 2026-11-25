# All-reduce + AttnRes + RMSNorm entry

## What it does
The shape-agnostic entry of the eight-rank all-reduce + AttnRes block mix + RMSNorm; a caller imports this package,
never a shape package. `attnres_comm` builds the communicator of the shape package that serves (hidden, world size):
`hidden` is a keyword (no activation exists yet), the world size comes from the group, and the package is imported on
that first call. The op is the communicator's method `allreduce_attnres_rmsnorm`, unchanged: numerics, CUDA graph
behaviour and warm-up are the shape package's. `allreduce_attnres_rmsnorm_blocker` takes the communicator and the
method's arguments and returns None or a one-line reason (no kernel for the communicator, a call it was not
built for, rows out of range, an operand layout the package rejects); it is host-only (no device read, no allocation,
safe inside graph capture), and the caller falls back on a reason.

## Entry points
- `available`: gfx950 with FlyDSL (the routed package's requirement).
- `attnres_comm`: the communicator of this rank (collective); `attnres_comm_blocker`: whether a key has one;
  `prepare_attnres_comm`: the communicator or the reason.
- `allreduce_attnres_rmsnorm_blocker`: per call, whether `comm.allreduce_attnres_rmsnorm` serves it.
- `allreduce_attnres_rmsnorm_shapes`: the covered (hidden, world size) keys, for load-time decisions.

Dispatch: `_ROUTES` (`amd_kernels.routes.Routes`) in `host.py`; a call routes by its communicator.

## Performance
The entry adds no code to the call (the method is the package's); the caller's per-call check is the reason. MI355X,
8 ranks, the pre-MLP call (prefix, 8 bank rows, output norm), us per call, max over ranks; graph: 20 calls a graph, min
of 15 rounds; host: Python time of an eager call, min of 50 batches of 100 (`benchmark_routing.py`).

| M | graph package | graph method | host package | host method | host reason |
|---|---|---|---|---|---|
| 1 | 8.58 | 8.58 | 15.26 | 15.29 | 4.01 |
| 64 | 12.85 | 12.83 | 15.30 | 15.31 | 4.02 |
| 256 | 30.99 | 31.00 | 15.20 | 15.27 | 4.00 |

## Tests
```bash
python3 -m pytest tests/norm/attnres_rmsnorm_common -q   # test_routing: no GPU; test_entry_ranks8: 8 GPUs
torchrun --nproc_per_node 8 benchmarks/norm/attnres_rmsnorm_common/benchmark_routing.py
```
