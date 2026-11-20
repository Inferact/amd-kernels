# MXFP4 MoE with MXFP8 shared expert, H6144 I1536 E128 top4 (gfx950)

## What it does

One TP2 rank's MoE block of MiniMax-M3.1-Flash, FlyDSL only: bf16x3 split-K router + biased top-4 of 128 experts
(weights include the routed scale 2.0), MXFP4 x MXFP4 routed experts (I 1536 per rank, SwiGLU-OAI: alpha 1.702,
limit 7, up + 1), MXFP8 shared expert (the rank's I 1536, from a shard copy or read in place from the full I 3072
copy). Caller: vLLM's MoE layer (decode graphs to 1024 rows, eager prefill).
Input: normed BF16 `[M, 6144]`, contiguous, M 1-87381. One quant rule at every M: MXFP4 with Quark-even e8m0 per 32
for the routed input and GEMM1 output, E4M3 with e8m0 `2^ceil(log2(amax / 448))` per 32 for the shared input and FP32
SwiGLU output; M picks only tiles, launches and sum order. Slot k = `topk_weights[:, k] * expert(x)` over the full
I 1536, rounded once to BF16; no output atomics, bitwise repeatable, graph replay = eager. Contract (`reference.py`):
`mxfp4_quant_ref`, `shared_mxfp8_quant_ref` bit for bit, `w4a4_moe_ref` within rel. L2 4e-3, `shared_mxfp8_ref`.
Router partials: K split and sum order follow M and the router kernel (weight err <= 5.1e-7). The one-launch GEMMs
share one per-device ready-counter arena, so two calls must not run concurrently on one device.

## Entry points

- `MoeWeights.from_row_major`: the router and routed experts' checkpoint tensors (and the `SharedWeights`, which
  `mxfp4_moe` needs) in the kernel layouts, plus a 3 MiB line-layout gate copy; once, before graph capture.
- `SharedWeights.from_row_major`: the shared expert's MXFP8 weights and scales as a bit-exact permutation (AITER's
  preshuffle): a TP2 shard, or the full expert with the TP rank whose shard the BF16 entry computes (`SHARED_FULL`).
- `unsupported_reason`: why a checkpoint's MoE tensors or activation do not fit (None: they fit).
- `mxfp4_moe`: the MoE block of a normed BF16 input or a prepared input: reduced, BF16 `[M, 6144]` (slots 0..3, then
  shared, summed in FP32), or the partials (slots `[M, 4, 6144]`, shared `[M, 6144]`) for a fused slots collective.
  An `amd_kernels.warm_up` entry: the first call of a mode compiles its variants at every `chain.WARM_UP_TOKENS` count.
- `warm_up_mxfp4_moe`: at start-up, loads every variant of a mode (`sp`: the SP modes too); one layer serves all.
- `prepare_mxfp4_moe`: routes and quantizes the local SP shard into a `MoeInput`; `MoeInputCompact` (7328 B/token)
  replaces the FP4 codes by a 1-bit mask; `MoeInput.routed` (3296 B/token) gives the routed slots only.
- `gather_mxfp4_moe`: packs and gathers the compact (or with `routed_only` the routed) payload by a warmed
  `AllGatherComm` of `world` ranks (default 2; at most 32768 gathered rows: 16384 per rank at two ranks).
- `shared_expert_full`: the full shared expert (I 3072) of prepared MXFP8 rows, on a side stream if asked (SP: the
  rows' owner adds it in its reduce-scatter).
- `set_dbo_query`: the server's query whether a call runs inside a DBO micro-batch (vLLM:
  `vllm.v1.worker.ubatching.dbo_enabled`); inside one the shared expert stays on the caller's stream.

Dispatch is by M only, also under graph capture: `chain.py` picks the router, sort and expert host
(`experts_a4w4_small.py`, `experts_a4w4_bm128.py`, `experts_a4w4.py`), each host its kernels.

## Performance

One MI355X, synthetic weights (`benchmark_shared_moe.py --baseline`), shared expert on both sides: reduced entry vs
the block vLLM runs at TP2 without it (router GEMM + AITER biased top-k + AITER W4A4 `fused_moe` + Triton MXFP8 shared
MLP on a side stream), same inputs, same process; graph replay of 100 inputs (16 above 2048 rows), median of 7 x 10
replays, us. Last row: the partials mode. Repeated A/Bs at 384-640 rows: -1.3..-10.5%.

| M | 8 | 32 | 64 | 128 | 256 | 320 | 384 | 448 | 512 | 640 | 1024 | 2048 | 4096 | 8192 | 32768 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 97.3 | 215.8 | 294.5 | 327.1 | 351.0 | 357.6 | 364.2 | 377.3 | 373.6 | 401.2 | 437.4 | 537.9 | 821.6 | 1366.7 | 5362.5 |
| this entry | 79.9 | 189.9 | 253.8 | 301.1 | 324.6 | 342.7 | 353.9 | 364.4 | 372.7 | 395.2 | 407.7 | 451.7 | 656.6 | 1005.7 | 3256.0 |
| change | -17.9% | -12.0% | -13.8% | -8.0% | -7.5% | -4.2% | -2.8% | -3.4% | -0.2% | -1.5% | -6.8% | -16.0% | -20.1% | -26.4% | -39.3% |
| partials | 78.4 | | 254.6 | | 317.3 | | 343.0 | | 359.9 | | 402.1 | | 625.4 | | 2981.8 |

## Tests

```bash
python3 -m pytest tests/moe/mxfp4_h6144_i1536_e128_top4 -q  # test_gather_ranks2.py: 2 GPUs
python3 benchmarks/moe/mxfp4_h6144_i1536_e128_top4/benchmark_shared_moe.py --baseline  # entry vs the vLLM TP2 block
# also: benchmark_prepared.py, benchmark_sidecar.py, benchmark_baseline_moe.py (the baseline alone)
```
