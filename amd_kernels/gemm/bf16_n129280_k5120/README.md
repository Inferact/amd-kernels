# bf16 GEMM, N 129280 x K 5120 (the lm_head, vocabulary replicated per rank)

## What it does
`out [M, 129280] = x [M, 5120] @ W^T`, bf16 operands, fp32 sums, one bf16 rounding: the full lm_head of a DP rank
(`lm_head.weight` bf16 [129280, 5120], 1.32 GB, read once per call). One tensor serves the target lm_head (verify rows)
and the draft lm_head (draft rows); M is any row count, served at the graph pad sizes 16 to 768. W layout: `row` (the
checkpoint) or `shuf` (a bit-exact permutation once at load, so one DMA reads 1 KB in one piece); both give the same
bits. No split-K, no partials, no output atomics, no bias. Contract (`reference.py`): the fp64 product rounded once to
bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3, deterministic; a row's bits do not depend on M within one kernel
(up to 64 rows, 65-256 rows, above 256 rows); NaN / inf propagate as in an fp32 sum. The output is the full logits tensor: the
sampler reads whole rows (the Markov draft adds a bias to every logit before the argmax and caches the sum), so no
reduction is fused into the epilogue.

## Entry points
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout; replace the parameter once
  before graph capture, keep the target / draft alias, free the row-major copy.
- `bf16_gemm_n129280_k5120`: the logits of any row count (launches of 2048 rows), graph-safe, no host sync. An
  `amd_kernels.warm_up` entry: the first call of each layout compiles its 16 kernel variants.
- `warmup_bf16_gemm_n129280_k5120`: the same warm-up, explicitly.
- `m_tile_fragments`, `m_rows`, `k_starts`: the M tiling and the K start rule of the 8-wave launches.
- `supports`: whether a weight is the [129280, 5120] bf16 contiguous tensor and there is no bias.

`host.py` picks the kernel and M tile by row count: `kernel.py` up to 64 rows, `large.py` above.

## Performance
One MI355X (shared node, so best of 7 rounds). Graph replay, the real-size weight (1.32 GB, above the 256 MB MALL: HBM
on every call), `shuf`, 100 distinct activation sets, rounds A B B A, best us. Baseline = vLLM's
`rocm_unquantized_gemm_impl` (`wvSplitKrc` 10-128 rows, hipBLASLt above). Roofline: 1.33 GB at 8 TB/s = 166 us; MFMA at
2.5 PF is 8.5 us per 16 rows.

| M | 16 | 32 | 64 | 96 | 128 | 160 | 192 | 256 | 384 | 512 | 640 | 768 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM | 191.6 | 197.9 | 209.0 | 210.5 | 252.5 | 242.0 | 248.8 | 299.9 | 439.5 | 580.9 | 663.2 | 804.3 |
| this | 202.3 | 204.4 | 209.1 | 215.5 | 219.7 | 239.3 | 270.6 | 318.1 | 471.9 | 554.4 | 735.1 | 808.3 |
| TB/s | 6.54 | 6.48 | 6.33 | 6.14 | 6.03 | 5.53 | 4.89 | 4.16 | 2.81 | 2.39 | 1.80 | 1.64 |

Ahead at 128 and 512 rows, level at 64, 160 and 768, 3-10 % behind elsewhere (vLLM's rows 96-128 jump is the switch to
hipBLASLt). Limit: up to 128 rows the HBM weight stream, 6.8 TB/s measured cold; achieved 6.0-6.5 TB/s (89-96 %). From
192 rows the global -> LDS DMA of one 8-wave workgroup per CU: the DMAs alone take 640 us at 768 rows, achieved 808 us
(79 %; 1.26 PF, 56 % of the 2.25 PF bf16 MFMA ceiling).

## Tests
- `python3 -m pytest tests/gemm/bf16_n129280_k5120 -q`
- `python3 benchmarks/gemm/bf16_n129280_k5120/benchmark_gemm.py [M ...]`
- `python3 benchmarks/gemm/bf16_n129280_k5120/sweep_rings.py [mt ...]` (the `large.RINGS` sweep)
