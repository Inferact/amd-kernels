# Markov draft chain, 5 steps, vocab 129280, rank 256, every request of the batch in one launch

## What it does

The sequential sampling of a DSpark speculator (`draft_sample_method=probabilistic`, bf16 logits, fp32 Gumbel draws,
no confidence head, no reduced draft vocabulary, replicated draft-logits cache) for all `n` requests of a rank and its
5 draft steps. Per request r and step s, from the previous token p (step 0: `input_ids[anchor_idx[r]]`, then the
draft token of step s - 1): `bias = bf16(markov_w1[p] . markov_w2^T)` (fp32 sum over rank 256), `y = bf16(base[r, s] +
bias)` is stored as the draft-cache row `cache[slot, s]` (pre-temperature; nothing for slot < 0), and the token is the
lowest column of the maximum of `y` at temperature 0, else of `y / T + noise`, noise = vLLM's murmur3 Gumbel draw of
(`seed ^ (s << 32)`, `sample_pos[r, s] - 1 + 2^30`, vocab id). The draws are bit for bit vLLM's `gumbel_sample` (the
tests compare them with its Triton kernel). The bias sum has the MFMA's order, so a bias can differ from another
order's in its last bf16 bit (`reference.py` is the contract and states what the tests compare). The previous token
of a step depends on the argmax of the whole vocabulary, so the logits cannot be reduced inside the lm_head GEMM:
this entry reads the bf16 base logits `[n * 5, 129280]` the GEMM wrote, and writes the cache rows the rejection
sampler reads. 253 workgroups (one per CU) each keep a 512-column tile of `markov_w2` in registers across the five
steps and swap tile winners through memory (tagged entries, no atomics, no resets); `kernel.py` documents the
protocol. It needs all 253 CUs free while it runs: its workgroups wait for each other. A token is always in `[0, V)`:
a NaN or +inf logit never wins, a row with some finite logit takes the lowest column of the finite maximum, and a row
without a winner (every value NaN or +-inf at some step) gets token 0 (`reference.py` states it).

## Entry points

- `markov_chain`: the chain for any request count (launches of 160 rows above that); int32 / int64 slot map and
  token ids, strided base rows; deterministic, graph safe, no host sync. An `amd_kernels.warm_up` entry: the first
  call compiles the 7 row-count variants (16 .. 160 rows).
- `warmup_markov_chain`: the same warm-up and the workspaces, explicitly, before graph capture.
- `markov_chain_reference`: the whole chain in torch (the contract); `nrf_for`: the variant of a row count
  (`host.py`, the dispatch); `available`: gfx950 with FlyDSL importable.

## Performance

One MI355X. Graph replay, 100 distinct bf16 base-logit sets per graph (`n = 8 .. 160` rows, 5 steps each), five
alternated rounds in the order baseline, entry, entry, baseline, median µs per chain. Baseline = vLLM's per-step
loop in `DSparkSpeculator._sample_markov_chain`: `markov_w1` embedding, bias GEMM (`F.linear`, hipBLASLt), `gumbel_sample`
with the fused bias (Triton partials, combine, cache write) and the draft-token copy, 5 times (about 9 kernels a step).

| requests | 8 | 16 | 32 | 48 | 64 | 96 | 128 | 160 |
|---|---|---|---|---|---|---|---|---|
| vLLM loop | 179.5 | 218.1 | 220.1 | 245.6 | 280.2 | 331.6 | 394.8 | 472.1 |
| this entry | 51.1 | 53.9 | 79.6 | 109.3 | 132.5 | 190.6 | 242.5 | 300.7 |

Above about 32 requests the noise (about 70 vector instructions a logit) bounds it, not memory.

## Tests

```bash
python3 -m pytest tests/sampling/markov_chain_gumbel_bf16_v129280_r256_s5 -q
python3 benchmarks/sampling/markov_chain_gumbel_bf16_v129280_r256_s5/benchmark_markov_chain.py [n ...]
```

Degenerate rows (all NaN, all -inf, mixed, one finite column, +inf; T 0 and 1): `test_degenerate_rows_stay_in_range`.
