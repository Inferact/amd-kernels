# Standard rejection sampling, 5 draft tokens, vocab 129280: two launches

## What it does

Verifies the draft tokens of a speculative-decoding step against the target distribution, as vLLM's
`rejection_sample` does for `rejection_sample_method=standard` (Leviathan et al.; replicated draft-logits cache, no
watermark, block verification or FLy window): `B` requests x 6 logits rows (5 draft steps and the bonus row), target
rows `[B * 6, 129280]` and the bf16 pre-temperature draft rows `[R, 5, 129280]`. Per request: the draw `u` is Triton's
Philox `tl.rand(seed, pos)`; temperature 0 accepts a draft that is the target's lowest-id argmax, otherwise
`log p > log u + log q`; the first rejected row (or the bonus row) gets one token, the argmax of the residual
`max(p - q, 0)` (the target row for the bonus) plus Gumbel noise from vLLM's murmur3 draw; temperature 0 takes the
target argmax. `synthetic_rates` swaps the ratio test for `u < rate`, as the synthetic mode. Contract (`reference.py`):
the same rule in fp64; the kernel's fp32 sums and hardware exp / log differ in the last bits, so a decision can differ
only where its margin is below ~1e-6 (tests compare tokens where it exceeds 1e-3); two runs match bit for bit, a
request does not depend on the others. Returned ids are in `[0, V)` or the -1 pad for any input (a row with no
winner, all NaN or all -inf, resamples token 0). Equal to vLLM's Triton path on 192 mixed-temperature requests, fp32
and bf16 target (that check needs vLLM, outside the tests).

Producer and consumer: the target rows are the bf16 output of the lm_head GEMM
([`gemm/bf16_n129280_k5120`](../../gemm/bf16_n129280_k5120/README.md), `[6B, 129280]`), taken raw: the kernel divides by the
request's temperature (fp32, as vLLM's `apply_temperature`), so no fp32 copy or temperature launch sits between them
(fp32 rows with the temperature, top-k / top-p mask already applied are also taken). The draft rows are the bf16 cache the
Markov sampler writes ([`sampling/markov_chain_gumbel_bf16_v129280_r256_s5`](../markov_chain_gumbel_bf16_v129280_r256_s5/),
`[slot, step, 129280]`, pre-temperature). The tokens `[B, 6]` and counts `[B]` go to the runner.

## Entry points

- `rejection_sample`: the sampler (tokens `[B, 6]`, -1 after the resampled token, and the sampled count `[B]`);
  graph safe; one workspace per device, so launches go on one stream. An `amd_kernels.warm_up` entry: the first call of
  a mode (target dtype, ratio test or synthetic) compiles it.
- `warmup_rejection_sample`: compiles the four modes and allocates the workspace, before graph capture.
- `rejection_sample_reference`: the torch contract, with the per-request decision margin. `available`: gfx950 with
  FlyDSL importable. `V`, `NSPEC`, `ROWS`: 129280, 5, 6.

`host.py` is the dispatch: block statistics of the 5 verified rows (the one pass over the logits), then one
workgroup per (request, block) that decides, scans the resample row and combines through a ticket (`kernel.py`).

## Performance

One MI355X, graph replay of 6-95 distinct inputs (> 2 GB), draft rows noisy copies of the target rows, temperatures
0 / 0.7 / 1, rounds A B B A, median us per `rejection_sample` call. bf16 rows: A = what vLLM runs on the lm_head output
(fp32 copy, `apply_temperature`, the 4 Triton launches of `rejection_sample`); fp32 rows: A = `rejection_sample` alone.

| Requests | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| bf16 rows: vLLM chain | 45.9 | 65.5 | 102.9 | 171.0 | 312.5 |
| bf16 rows: this entry | 21.3 | 25.6 | 38.5 | 63.6 | 112.9 |
| change | -54% | -61% | -63% | -63% | -64% |
| fp32 rows: Triton path | 31.8 | 45.1 | 67.3 | 108.2 | 175.6 |
| fp32 rows: this entry | 22.1 | 26.4 | 41.5 | 70.5 | 131.3 |
| change | -31% | -42% | -38% | -35% | -25% |

## Tests

```bash
python3 -m pytest tests/sampling/rejection_standard_v129280_s5 -q
python3 benchmarks/sampling/rejection_standard_v129280_s5/benchmark_rejection.py [--target fp32] [--reqs 8 32]
```

Degenerate rows (all NaN, all -inf): `test_degenerate_rows_return_in_range_ids`.
