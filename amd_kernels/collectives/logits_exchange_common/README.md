# lm_head shard GEMM + logits exchange entry (vocabulary-sharded logits rows to their owners or to every rank)

## What it does

The shape-agnostic entry of the lm_head exchange: the factory reads the shape from keyword dims (vocabulary shard N,
hidden K) and the group (world size), builds the exchange package's communicator, imports the GEMM package that serves
the shard (both on first use) and composes them in a `LogitsExchange` with every first-use buffer taken, so no call
allocates (graph capture). `gemm_exchange` runs the [N, K] shard GEMM of the rows `x[row_indices]` and pushes its
output to the owners (`counts`, rows sorted by owner) or to every rank (`counts=None`): the GEMM pushes the rows itself
up to the crossover the exchange package measured, above it writes them locally and the exchange's push moves them;
`recv` reads them. Computation, dtypes and numerics are the packages' (their `reference.py`): bf16, one rounding of an
fp32 sum, an exact byte exchange.

Routed: (20480, 7168, 8) -> `collectives/all_to_all_bf16_n20480_ranks8` + `gemm/bf16_n20480_k7168`. K is in the key:
`gemm_exchange` runs the GEMM of one weight shape. A call's reason routes by the communicator's key and compares the
weight with it; it is host-only and takes the method's arguments.

## Entry points

- `available`: gfx950 with FlyDSL importable.
- `logits_gemm_exchange_shapes`: the covered (N, K, world), the load-time check of an lm_head shard.
- `logits_exchange_comm`, `logits_exchange_comm_blocker`, `prepare_logits_exchange_comm`: the collective
  communicator, a `LogitsExchange` (two receive buffers, or one when every rank reads an exchange before any issues
  the next), and the communicator or the reason it cannot be built.
- `LogitsExchange.gemm_exchange`, `.recv`, `.warmup`, `.close`: the call the model makes, the in-place rows, the
  collective compile before capture and the collective teardown.
- `logits_gemm_exchange_blocker`: whether `comm.gemm_exchange(...)` with these arguments is served.

Dispatch: `host._ROUTES` and `host._GEMMS` ((N, K, world) -> the exchange host, the GEMM host and op and the
push crossovers).

## Performance

The entry adds no kernel: `LogitsExchange.gemm_exchange` against the two packages called directly in the same order.
Eight MI355X, all-gather mode, max over the ranks, min of 15 rounds, us per call: graph replay of 24 calls, host time
of an eager call (enqueue) and of the reason (`benchmark_routing.py`).

| rows | graph, packages | graph, entry | host, packages | host, entry | reason |
|---|---|---|---|---|---|
| 8 | 54.15 | 54.62 | 20.31 | 19.70 | 1.43 |
| 64 | 111.72 | 111.96 | 31.70 | 30.44 | 1.42 |
| 256 | 311.32 | 310.08 | 32.45 | 30.98 | 1.41 |

## Tests

- `python3 -m pytest tests/collectives/logits_exchange_common -q` (routing on CPU, `test_entry_ranks8.py` on 8 GPUs)
- `python3 benchmarks/collectives/logits_exchange_common/benchmark_routing.py` (8 GPUs)
