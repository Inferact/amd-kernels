# bf16 GEMM entries: latent down projection + MXFP4 quant, N-sharded linear + gather + RMSNorm

## What it does

One shape-agnostic entry per op: it reads the shape from the operands, calls the shape package that serves it
(imported on first use) and, for a call no package covers, says why in a one-line reason so the caller falls back.
The computation, dtypes and numerics are the routed package's (its `reference.py`); the entry passes the arguments
through unchanged, so its outputs are the package's bits.

- `bf16_latent_down_mxfp4_quant`: (codes, scales) = MXFP4 of bf16(x [M, K] @ W^T), key (N, K) of the weight;
  `gemm/bf16_latent_down_mxfp4_quant_k7168_n3584`.
- `bf16_linear_gather`: RMSNorm of the full [rows, N shard x world] projection on every rank of a group, from the
  packed activations and this rank's packed weight shard, key (N shard, K, world);
  `gemm/bf16_linear_gather_n896_k35840_ranks8`, 1-256 rows.

A reason function is host-only (shapes, dtypes, strides, ints; no device read, no allocation) and takes exactly the
arguments of its op. The packed weights keep the raw weights' shapes: pass the entry's `pack` output.

## Entry points

- `available`: gfx950 with FlyDSL importable.
- `bf16_latent_down_mxfp4_quant`, `..._blocker`: the op and whether a call is served.
- `bf16_latent_down_mxfp4_quant_shapes`, `..._weight_reason`: load-time coverage, from the weight (and bias) alone.
- `latent_down_xcd_local`: turns the L2-local partial exchange off on a device (a kernel running beside it).
- `latent_down_swap_stream_workspaces`: lend one stream's workspaces to another (a graph capture stream).
- `linear_gather_comm`, `linear_gather_comm_blocker`, `prepare_linear_gather_comm`: the collective communicator of
  the group, whether it can be built, the communicator or the reason.
- `linear_gather_pack_weight`, `linear_gather_pack_aux` (+ `_blocker`), `linear_gather_packed_tiles`,
  `linear_gather_rows_buffer`: the load-time weight layout and the packed activations.
- `bf16_linear_gather`, `..._blocker`, `..._shapes`: the op (routed by its communicator) and its coverage.

Dispatch: `_ROUTES` in `latent_down.py` and `linear_gather.py` (key -> host module of the shape package).

## Performance

The entry adds only the dispatch (one cached lookup per call); a CUDA graph holds the same kernels. One MI355X (latent
down) / eight (linear gather, max over ranks), min of 15 rounds, us per call: graph replay of 24 calls, host time of
an eager call (enqueue), the routing alone (dispatch) and the reason (`benchmark_routing.py`).

| op, rows | graph direct | graph entry | host direct | host entry | dispatch | reason |
|---|---|---|---|---|---|---|
| latent down, 1 | 9.58 | 9.75 | 10.20 | 10.21 | 0.17 | 2.25 |
| latent down, 64 | 12.74 | 12.70 | 10.12 | 10.34 | 0.17 | 2.22 |
| latent down, 192 | 23.76 | 23.72 | 10.03 | 10.32 | 0.17 | 2.28 |
| linear gather, 1 | 15.44 | 15.52 | 16.62 | 15.91 | 0.07 | 2.09 |
| linear gather, 40 | 21.34 | 21.37 | 16.65 | 16.74 | 0.08 | 2.08 |
| linear gather, 192 | 47.32 | 47.26 | 16.70 | 16.62 | 0.07 | 2.12 |

## Tests

- `python3 -m pytest tests/gemm/bf16_common -q` (routing on CPU, latent down on 1 GPU, `test_linear_gather_ranks8.py`
  on 8 GPUs)
- `python3 benchmarks/gemm/bf16_common/benchmark_routing.py --op latent_down` (1 GPU), `--op linear_gather` (8 GPUs)
