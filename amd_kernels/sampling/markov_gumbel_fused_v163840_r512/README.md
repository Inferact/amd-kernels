# Fused Markov draft chain of a Kimi-K3.1 DSpark draft, vocab 163840, Markov rank 512

## What it does

The loop of vLLM's `DSparkSpeculator._sample_sequential` after the base logits, for `draft_sample_method=probabilistic`
(draft logits cached for rejection): per step s and request r, `bias = bf16(w1[prev] . w2^T)` (fp32 accumulation, one
bf16 rounding: the GEMV), `logits = bf16(base[r, s] + bias)` stored into the draft logits cache column s (honouring
`DraftLogitsCache.row_mapping`), then vLLM's Gumbel draw (murmur3 uniform of (seed ^ (s << 32), sample_pos - 1 + 2^30,
column), `-log(-log1p_neg_stable(u))`, `logits / T` with IEEE division, T = 0 plain argmax, invalid rows plain argmax and
no cache write) and the argmax (first column on ties) into `draft_tokens[r, s]`, which is the next step's `prev`. One
persistent launch for all k steps in place of the stock ~9 launches a step: 256 workgroups (one per CU, all resident) of
four waves stream the 168 MB `markov_w2` once a step for all rows, synchronise per step with agent-scope records (no
atomics into outputs), keep 5-7 weight tiles a workgroup in LDS and alternate the sweep direction so that the last tiles
read are still in L2. Shapes: 1-48 requests, 1-8 steps, bf16 `markov_w1` / `markov_w2` [163840, 512] contiguous. Caller:
the K3.1 DSpark speculator at c16-c48 (8 / 13 / 20 / 48 requests, 4 / 4 / 3 / 3 steps). Contract (`reference.py`): bit for
bit vLLM's `gumbel_sample` on the cached logits (434 draws against the real op, 0 mismatches); the bias sum order differs
from the GEMV's (a few logits one bf16 ulp); inputs finite.

## Entry points

- `markov_gumbel_fused`: all k steps over `base` [n, k, 163840] into `draft_tokens` and the cache, graph safe, warms its
  three row-tile variants up on its first call. The 256 workgroups must be co-resident: nothing else may hold CUs of the
  device while it runs (a wait that never ends is abandoned after about 3 s: wrong draft tokens, not a hung GPU).
- `scratch_dwords`: size of the int32 scratch (generation, results, rank counters, records), zeroed once, never reset.
- `available`: gfx950 with FlyDSL importable.

## Performance

One MI355X, graph replay, 3 rotated `markov_w2` copies, 8 input sets (HBM-cold), T 1.0, median of alternated rounds, us per
chain. Ceiling A = k launches of the pure `markov_w2` stream (the floor of one launch per step); ceiling B = one
persistent launch of k stream passes without step synchronisation; both plus the base read / cache write at 6.1 TB/s
(`skeleton.py`).

| requests / steps (conf) | 8 / 4 (c16) | 13 / 4 (c24) | 20 / 3 (c32) | 48 / 3 (c48) |
|---|---|---|---|---|
| vLLM loop | 238.1 | 286.3 | 202.0 | 305.1 |
| this entry | 109.1 | 112.8 | 95.8 | 115.1 |
| speedup | 2.18x | 2.54x | 2.11x | 2.65x |
| % of ceiling A (115.1 / 117.1 / 89.6 / 98.6 us) | 105% | 104% | 94% | 86% |
| % of ceiling B (100.3 / 102.3 / 77.7 / 86.6 us) | 92% | 91% | 81% | 75% |

T 0.7: 111.3 / 114.4 / 96.4 / 120.2 us; T 0 (no noise): 105.9 / 108.4 / 90.9 / 105.9 us. Per step (timestamps,
`timeline.py`): 0.8 us prologue (tokens, `w1` rows, operand), the tile loop (20.8 us median at 8 rows), 2-5 us
between the first and the last workgroup finishing, 2.7-3.8 us for the two cross-XCD hops of the argmax (0.54 us a
hop measured) and the broadcast.

## Tests

```bash
python3 -m pytest tests/sampling/markov_gumbel_fused_v163840_r512 -q
python3 benchmarks/sampling/markov_gumbel_fused_v163840_r512/benchmark_markov_gumbel_fused.py [--temp 0.7] [--rounds 9]
python3 benchmarks/sampling/markov_gumbel_fused_v163840_r512/timeline.py 8 4   # per-step timestamps
```
