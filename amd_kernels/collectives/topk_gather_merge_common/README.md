# DCP top-k score gather + merge entry (gfx950, multi-rank)

## What it does
The shape-agnostic entry of the DSA decode's score all-gather + global top-k merge + this rank's CSR KV slots: the
factory reads top-k from its keyword and the world size from the group, builds the communicator of the shape package
that serves them, and for a configuration no package serves returns a reason, so the caller keeps its fallback. A
caller imports no shape package. Computation, dtypes and numerics are the routed package's (its README and
`reference.py`); the caller calls the communicator itself, so its outputs are the package's bit for bit.

| key (top-k, world size) | shape package |
|---|---|
| (2048, 8) | `collectives/dcp8_topk_gather_merge_top2048` |

## Entry points
- `available`: gfx950 with FlyDSL, the load-time check before any shape question.
- `topk_gather_merge_comm`: the collective factory (every rank calls it); the communicator is called per step,
  `.warmup()` before capture and `.close()` after its graphs are retired.
- `topk_gather_merge_comm_blocker`: the factory's arguments -> None or a one-line reason.
- `prepare_topk_gather_merge_comm`: the communicator, or the reason it cannot be built.
- `topk_gather_merge_shapes`: the covered (top-k, world size), for a caller that decides at load.
- `topk_gather_merge_blocker`: the communicator and its call's arguments -> None or a one-line reason
  (host checks only, safe inside graph capture); it routes by the communicator's type and compares the call's top-k
  with the one it was built for.

Dispatch: `host._ROUTES` (module names, imported on first use).

## Performance
The entry adds nothing to a call: the caller calls the package's communicator. Eight MI355X, graph replay of 100
calls on distinct inputs, MAX over ranks, min of 15 rounds (each graph timed twice, interleaved), us per call.

| rows | communicator from the package | from the entry | added |
|---|---|---|---|
| 12 | 13.45 | 13.54 | +0.10 |
| 65 | 17.24 | 17.39 | +0.15 |
| 192 | 23.54 | 23.50 | -0.03 |

The call's reason takes 2.2 us of host time.

## Tests
```bash
python3 -m pytest tests/collectives/topk_gather_merge_common -q   # routing (host only); 8 GPUs: equal to the package
python3 benchmarks/collectives/topk_gather_merge_common/benchmark_routing.py   # 8 GPUs
```
