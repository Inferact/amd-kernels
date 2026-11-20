# Two-rank pushes on the SDMA engines, ordered by the GPU stream

## What it does

`SdmaPush` moves bytes between the two ranks of a GPU pair with a copy (SDMA) engine instead of CU kernels: no
workgroup reads the peer or spins on a flag, so a hand-off can run next to another stream's or DBO microbatch's compute
without taking its CUs. The GPU orders it: the copies start once the caller's stream reaches the push, and the
receiver's stream waits with a command-processor wait. The host blocks only to recycle a signal-ring slot `RING` (32)
pushes later. Bytes are copied unchanged (no arithmetic, no `reference.py`). Caller: `SdmaSeqParallelHandoffs`
(`collectives/seq_parallel_handoffs_h6144_q4096_ranks2`), the DBO microbatches of the two-rank SP prefill; push / wait
take the signatures of `sdma_push_ranks4`.

## Entry points

- `amd_kernels.collectives.sdma_push_ranks2.SdmaPush`: a pair's receive regions (one IPC pool) and flag words, built
  collectively on a gloo group.
- `SdmaPush.push`: copies this rank's items into the peer's region once the stream reaches the push.
- `SdmaPush.wait`: the stream waits until the peer's push of a sequence number has landed.
- `SdmaPush.drain`: the host waits for every push's copies and drops the held sources.

Eager only; both ranks push and wait on a region in the same order, the source and region lifetimes are the caller's
(`transport.py` docstring, which also holds the engine policy and the per-push steps).

## Performance

One xGMI link per pair: one engine reaches 61.2 GB/s per direction with both directions busy at 256 MB; 2 to 8
engines striped to the one peer gain <= 1% one way and lose up to 1.3% both ways (`bench_engines.py`), so `ENGINES` is
one engine. Push + wait of an SP MoE layer's hand-offs (R = T / 2 rows x 30176 B, both ranks at once, eager) vs RCCL
all-gather of the same bytes, us; a bf16 GEMM loop on another stream slows by at most 1% while the pushes run:

| T (tokens) | MB per rank | `SdmaPush` | RCCL all-gather |
|---:|---:|---:|---:|
| 4096 | 61.8 | 1042.5 | 1165.6 |
| 8192 | 123.6 | 2044.3 | 2279.8 |
| 16384 | 247.2 | 4057.9 | 4510.5 |
| 32768 | 494.4 | 8082.3 | 8942.3 |

## Tests

```bash
python3 -m pytest tests/collectives/sdma_push_ranks2 -q                       # 2 GPUs
python3 benchmarks/collectives/sdma_push_ranks2/bench_engines.py 8 64 256
python3 benchmarks/collectives/sdma_push_ranks2/benchmark_push.py
```
