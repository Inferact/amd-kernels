# Top-p mask of fp32 logits rows, vocab 163840: vLLM's split top-p chain in one launch

## What it does

Masks the columns outside the top-p nucleus of each row with -inf, in place, bit for bit what vLLM's
`apply_top_k_top_p_triton(logits, None, p)` does at <= 64 rows: its split top-p pipeline (`_topp_sb_stats_kernel`,
five `_topp_sb_step_kernel` rounds, `_topp_sb_mask_kernel`, Triton 3.8 on gfx950), seven launches and about twice
the time of this one. Per row (p < 1): S slices (32 at <= 8 rows, 16 above, by vLLM's
`_topp_split_count` on 256 CUs) give the max, the softmax sum and the min; five rounds of an 8-pivot geometric ladder
bracket the probability where the kept mass reaches p; the mask keeps `prob > pivot` or the max, and boundary
duplicates in column order. Contract (`reference.py`): the same fp32 operations in the same order (Triton's
`tl.exp` / `tl.log` expansions, each `tl.sum`'s butterfly, the two products the compiler fuses), compared exactly.
Rows with p >= 1 stay as they are. Bit equality with the Triton chain needs vLLM, so it is checked outside this
repository (row counts 1-64; Gaussian and LM-like rows at T 0.6-1.2, integer ties, -inf columns, p 1 rows, flat rows:
0 mismatching elements). Caller: the Kimi-K3.1 TP8 sampler (target top-p; with `--enable-batch-sharded-sampling` each
rank holds at most 24 rows at M = 192).

## Entry points

- `top_p_mask`: masks the columns outside each row's top-p nucleus with -inf, in place (fp32 `[rows, 163840]`, 1-64
  rows); one workspace per device (as vLLM's), so launches go on one stream; graph safe. An `amd_kernels.warm_up`
  entry: the first call compiles both slice counts.
- `top_p_blocker`: why the entry cannot stand in for a model's chain (None: it can).
- `warmup_top_p_mask`: compiles both variants and allocates the workspace; nothing compiles after it.
- `splits_for`: the slice count for a row count and CU count (vLLM's `_topp_split_count`); `host.py` dispatches on it.
- `top_p_mask_reference`: the torch contract. `available`: gfx950 with FlyDSL importable. `V` = 163840.

## Performance

One MI355X (shared with other jobs: minimum of rounds); graph replay of 40 distinct inputs (synthesized LM-like rows,
`--rows` for real ones), each a bf16 -> fp32 copy and the mask, copy subtracted; us per call; A = vLLM's Triton chain
(7 launches, `apply_top_k_top_p`), same inputs. Model of the critical path: seven dependent phases (stats, five pivot
rounds, mask), each a barrier among the row's S workgroups (measured 0.9 us at 8 blocks, 2.4 us at 64: 1.6 us at S = 32),
a reload of the 20 KB slice from L2 (0.1-0.25 us), and the ladder arithmetic (1-2 us): 7 x ~4.2 = ~30 us at one row;
the HBM time of a row (655 KB / 6.5 TB/s = 0.1 us) is not binding.

| Rows | 1 | 3 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|---|---|
| Triton chain | 54.1 | 65.9 | 71.7 | 72.8 | 101.7 | 104.3 | 150.7 | 155.9 | 204.2 | 251.9 |
| this entry | 30.1 | 38.3 | 42.3 | 43.0 | 52.8 | 54.6 | 66.4 | 70.1 | 107.7 | 124.7 |

## Tests

```bash
python3 -m pytest tests/sampling/top_p_mask_f32_v163840 -q
python3 benchmarks/sampling/top_p_mask_f32_v163840/benchmark_top_p_mask.py [--rows <logits rows .pt>]
```
