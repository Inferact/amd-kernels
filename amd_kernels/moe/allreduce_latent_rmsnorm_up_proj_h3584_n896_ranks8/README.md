# Latent all-reduce + RMSNorm + hidden-shard up projection, h3584 n896, eight ranks

## What it does
The MoE-layer tail of a Kimi-K3.1 decode step on TP8, in one launch per layer: the all-reduce of the routed experts'
latent partial (bf16 [M, 3584], 1..256 rows), `RMSNorm(3584)` of the reduced latent and the up projection of this
rank's 896-column hidden shard (bf16 weight [896, 3584], `up_proj.weight[896 r : 896 (r + 1)]`):
`out[M, 896] = shared[M, 896] + rmsnorm(sum_r latent_r) @ W^T`. `reference.py` is the contract: the eight partials
are added in rank order in fp32 and rounded once to bf16, `xn = bf16(latent * w_norm)` feeds the GEMM, `1 / rms` is
applied to the fp32 sum, one bf16 rounding at the end; the output is within `2^-8 |y| + 2^-8 max|up|` of the fp64
evaluation, deterministic, no atomic adds into outputs. `out` may alias `shared`: the stock layout (a view of the
TP-sharded shared-expert partial, then the final all-reduce adds the other ranks' shares) and the replicated layout
(`shared` = this rank's columns of the replicated shared output, `out` a zero-padded [M, 7168] partial buffer of
which only this rank's columns are ever written, so the final all-reduce, or an all-gather, of the eight buffers
is exact; pad with -0.0, `replicated_partial_buffer`, so even a real -0.0 survives bit for bit) use the same call.

## Entry points
- `LatentTailComm`: the communicator, a collective on the eight TP ranks, one per stream; `close()` is collective too.
- `allreduce_latent_rmsnorm_up_proj`: the call (an `amd_kernels.warm_up` entry); graph-safe, no host work.
- `allreduce_slots_rmsnorm_up_proj`: the same with the routed experts' 16-slot sum folded in (1..64 rows).
- `replicated_partial_buffer`: the -0.0-padded [M, 7168] partial buffer of the replicated layout.
- `reference.latent_tail_reference`: the contract.
The dispatch (row tier -> all-reduce workgroup rows, up-projection tiles) lives in `tail_kernel.py` (`rb_of`,
`nsub_of`, `groups_of`); the protocol in its module docstring.

## Performance
Eight ranks, graph replay, 24 rotating weight sets, slowest rank, us. Stock = AITER all-reduce + vLLM's Triton
norm/up-proj. Floor = launch 1.57 + 2 hops (0.7 flag + xGMI bytes at 430 GB/s) + x read + MFMA + drain (model in
`benchmark_tail.py`).

| rows | stock AR+up | fused | speedup | floor | floor / fused |
|---|---|---|---|---|---|
| 12 | 12.2 | 11.3 | 1.08x | 5.3 | 47% |
| 24 | 13.2 | 11.6 | 1.14x | 5.7 | 49% |
| 40 | 13.5 | 12.3 | 1.09x | 6.1 | 50% |
| 65 | 16.7 | 14.2 | 1.17x | 6.9 | 48% |
| 80 | 17.2 | 14.9 | 1.15x | 7.3 | 49% |
| 96 | 18.2 | 15.6 | 1.16x | 7.8 | 50% |
| 128 | 20.0 | 17.3 | 1.16x | 9.4 | 55% |
| 192 | 25.2 | 21.4 | 1.18x | 11.3 | 53% |

Each hop costs 1.5-2x its isolated primitive (flag round trip, 7-destination store acks, poll after barrier). The
slot-sum entry adds 1.4..2.6 us over the call on a summed partial, about 1 us less than a separate sum.

## Tests
```bash
python3 -m pytest tests/moe/allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 -q          # 8 GPUs (test_up_proj: 1 GPU)
python3 benchmarks/moe/allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8/benchmark_tail.py  # 8 GPUs
```
