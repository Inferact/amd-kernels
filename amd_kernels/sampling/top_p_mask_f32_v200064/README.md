# Top-p mask of fp32 logits rows, vocab 200064: vLLM's top-p without top-k, bit for bit

## What it does

Masks the columns outside the top-p nucleus of each row with -inf, in place, bit for bit what vLLM's
`apply_top_k_top_p_triton(logits, None, p)` does (Triton 3.8 on gfx950). At <= 64 rows that is its split top-p
pipeline (`_topp_sb_stats_kernel`, five `_topp_sb_step_kernel` rounds, `_topp_sb_mask_kernel`), here one launch: S
slices (32 at <= 8 rows, 16 above, vLLM's `_topp_split_count` on 256 CUs) give the max, the softmax sum and the min;
five rounds of an 8-pivot ladder bracket the probability where the kept mass reaches p. Above 64 rows it is the
one-program-per-row `_topk_topp_kernel`: tile-0 statistics, the softmax sum, the outliers (probability above a
Gaussian pivot) compacted in column order, up to 18 bisection rounds on their 4096-position chunks (or on every column
when the outliers hold less than p), then the mask in logit space; one workgroup per row, 1.4-3.4x faster (flat to
peaked rows, 65-640 rows). Both keep boundary duplicates in column order. Contract (`reference.py`): the same fp32
operations in the same order (Triton's `tl.exp` / `tl.log` expansions, `v_sqrt_f32` for `tl.sqrt`, each `tl.sum`'s
butterfly, the products the compiler fuses), compared exactly. Rows with p >= 1 stay as they are. Bit equality with
vLLM needs vLLM, so it is checked outside this repository (1-64 rows; 65-300 rows: LM rows at T 0.6-1.5, heavy tails,
integer ties, -inf columns, p 1 rows, flat, constant and all -inf rows). Caller: the M3.1 sampler (target top-p,
batch-sharded owned rows) at TP4 and TP2.

## Entry points

- `top_p_mask`: masks the columns outside each row's top-p nucleus with -inf, in place (fp32 `[rows, 200064]`); one
  workspace per device up to 64 rows (as vLLM's: launches on one stream), above a caller's buffer (vLLM's) or one
  per device; graph safe. An `amd_kernels.warm_up` entry; `host.py` dispatches on the row count.
- `top_p_blocker`: why the entry cannot stand in for a model's top-p (None: it can).
- `warmup_top_p_mask`: compiles every variant; nothing compiles after it.
- `splits_for`: the slice count of the split pipeline (vLLM's `_topp_split_count`); `rows_grid`: the workgroups (and
  buffer rows) above 64 rows.
- `top_p_mask_reference`: the torch contract. `available`: gfx950 with FlyDSL importable. `V` = 200064.

## Performance

One MI355X; graph replay of 100 distinct inputs (20 above 64 rows), each a bf16 -> fp32 copy and the mask, copy
subtracted; us per call; p 0.95. Up to 64 rows real M3.1 logits rows (`--rows`, T 0.85-1.15) against vLLM's split
chain (7 launches); above, LM-like rows (a N(0, 2) body, 64 heads at 6-18) against `_topk_topp_kernel`
(`benchmark_top_p_mask.py` for this entry; vLLM side measured outside the repository).

| Rows | 1 | 8 | 16 | 32 | 64 | 80 | 128 | 160 | 256 | 416 | 640 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| vLLM | 52.0 | 77.7 | 125.1 | 179.8 | 280.5 | 426.3 | 444.9 | 445.8 | 472.1 | 865.5 | 1258.7 |
| this entry | 28.4 | 45.6 | 72.2 | 89.2 | 142.1 | 135.6 | 167.9 | 162.8 | 215.7 | 377.5 | 566.7 |

## Tests

```bash
python3 -m pytest tests/sampling/top_p_mask_f32_v200064 -q
python3 benchmarks/sampling/top_p_mask_f32_v200064/benchmark_top_p_mask.py [--rows <logits rows .pt>]
```
