# bf16 skinny linear: the dense decode GEMMs on one tile table

## What it does
`y [M, N] = x [M, K] @ W^T`, bf16 operands, fp32 sums, one bf16 rounding, M = 1..192 rows (more rows: more M tiles), for
the 15 dense weight shapes of a Kimi-K3.1 TP8 decode step (`SHAPES`: KDA in_proj 6288x7168, o_proj 7168x1536, DSA
qkv_a / q_b / g_proj / wq_index / wk_index / wv_index, latent down_proj 3584x7168, shared gate_up / down, layer-0 MLP,
lm_head shard 20480x7168, drafter context_proj 7168x35840 and context_kv 2880x7168). W is read from HBM in a load-time
packed layout (`pack_weight`, a permutation: every 16-byte lane load is one piece of a 1 KB run). Contract
(`reference.py`): the fp64 product rounded once to bf16; every element within one bf16 ulp (plus fp32 noise of
cancelling sums), per-row cosine >= 0.99999, deterministic bits (K split across workgroups is summed in split order by
the last workgroup of a tile, no output atomics). The tile (`wn`, `wk`, `ng`, `s`, depth, ring) comes from
`config.py`, measured per (shape, M tier) by graph-replay sweeps.

## Entry points
- `pack_weight`, `unpack_weight`: the load-time layout; replace the parameter once before graph capture, free the
  row-major tensor (the packed [N, K] bf16 tensor has the same shape and bytes).
- `bf16_linear`: `x [M, K] @ W^T` into `out`, graph-safe after the first call of a shape outside capture (an
  `amd_kernels.warm_up` entry: the first call compiles the 13 tile variants of the shape and allocates its split-K
  workspaces); one in-flight call per shape and device unless the caller passes its own `workspace`.
- `warmup_bf16_linear`: the same warm-up, explicitly. `supports`, `prefers_stock`: shape known / the stock op measured
  as fast as the kernel at that row count (`STOCK_FASTER`: small K at M >= 65-128, qkv_a, lm_head at M 65-128).
- Assumption: workgroup b runs on XCD b % 8 (the split-K tiles are exchanged through one XCD's L2; checked by
  `tests/.../test_xcd_placement.py`). `kernel.py` documents the tile; `host.py` maps M to a tier.

## Performance
One MI355X, graph replay, rotated weight copies (> 768 MB, cold HBM), minimum of 7 rounds, us per call: this kernel /
stock (vLLM `rocm_unquantized_gemm_impl`, AITER on: wvSplitKrc, hipBLASLt, AITER asm), (% of the floor). Floor =
1.57 us (empty graph kernel) + bytes / 6.9 TB/s (measured nontemporal HBM read). Bytes = W + x + y.

| N x K | op | M=12 | M=24 | M=40 | M=64 | M=80 | M=96 | M=128 | M=160 | M=192 |
|---|---|---|---|---|---|---|---|---|---|---|
| 6288x7168 | KDA in_proj_qkvgfab | 16.2 / 17.7 (91%) | 17.4 / 19.5 (85%) | 18.9 / 21.5 (78%) | 20.1 / 21.9 (74%) | 21.6 / 24.1 (69%) | 23.1 / 25.6 (65%) | 25.8 / 28.4 (59%) | 28.9 / 36.5 (53%) | 31.4 / 42.2 (49%) |
| 7168x1536 | o_proj (KDA, DSA) | 5.3 / 5.9 (90%) | 5.6 / 6.6 (86%) | 6.5 / 7.2 (75%) | 7.2 / 7.6 (68%) | 8.0 / 8.6 (62%) | 8.6 / 9.0 (59%) | 10.2 / 9.9 (50%) | 11.7 / 10.7 (44%) | 13.5 / 10.9 (39%) |
| 3584x7168 | latent down_proj | 10.3 / 11.6 (88%) | 10.8 / 11.9 (84%) | 11.7 / 12.9 (78%) | 13.2 / 14.6 (70%) | 14.1 / 17.6 (66%) | 15.1 / 18.4 (62%) | 16.9 / 20.6 (56%) | 19.6 / 29.1 (49%) | 22.6 / 31.0 (43%) |
| 1536x7168 | shared gate_up, g_proj | 6.4 / 7.0 (75%) | 6.8 / 7.4 (71%) | 7.6 / 8.2 (64%) | 8.5 / 9.1 (58%) | 9.4 / 10.0 (53%) | 10.3 / 10.5 (48%) | 11.9 / 11.8 (43%) | 14.0 / 13.4 (37%) | 16.3 / 16.4 (32%) |
| 2112x7168 | DSA fused_qkv_a | 8.1 / 8.1 (74%) | 8.8 / 8.4 (69%) | 10.4 / 9.0 (58%) | 11.7 / 10.1 (53%) | 13.0 / 10.9 (47%) | 13.8 / 11.6 (45%) | 15.4 / 12.7 (41%) | 17.8 / 16.9 (36%) | 21.2 / 20.9 (31%) |
| 7168x768 | shared down | 3.6 / 5.8 (89%) | 3.9 / 4.6 (83%) | 4.2 / 4.8 (77%) | 4.9 / 5.3 (68%) | 5.3 / 5.5 (63%) | 5.9 / 5.7 (57%) | 6.7 / 6.2 (52%) | 7.8 / 6.8 (45%) | 9.0 / 7.2 (40%) |
| 2304x1536 | DSA q_b | 3.6 / 5.6 (73%) | 4.2 / 5.6 (63%) | 4.7 / 5.0 (56%) | 5.1 / 5.6 (53%) | 5.6 / 5.6 (48%) | 6.0 / 5.8 (45%) | 6.8 / 6.2 (40%) | 7.9 / 6.4 (35%) | 9.1 / 7.2 (31%) |
| 4096x1536 | wq_index | 4.0 / 6.0 (85%) | 4.6 / 6.0 (75%) | 5.2 / 7.2 (66%) | 6.2 / 7.1 (56%) | 6.8 / 6.5 (52%) | 7.3 / 6.8 (49%) | 8.2 / 7.5 (44%) | 9.4 / 8.4 (39%) | 10.8 / 8.6 (34%) |
| 128x7168 | wk_index | 4.3 / 5.9 (43%) | 4.3 / 5.6 (44%) | 4.8 / 5.3 (40%) | 5.3 / 5.3 (37%) | 6.1 / 7.0 (33%) | 6.9 / 7.1 (30%) | 7.9 / 7.1 (27%) | 9.2 / 9.7 (24%) | 10.5 / 8.4 (21%) |
| 32x7168 | wv_index | 4.3 / 5.6 (39%) | 4.7 / 5.3 (36%) | 5.5 / 5.1 (31%) | 5.8 / 5.2 (30%) | 6.6 / 6.9 (27%) | 8.5 / 6.9 (22%) | 10.3 / 6.9 (18%) | 12.3 / 7.0 (16%) | 14.3 / 11.5 (14%) |
| 20480x7168 | lm_head shard | 43.9 / 43.7 (101%) | 45.2 / 44.5 (98%) | 46.2 / 45.8 (96%) | 51.0 / 50.2 (88%) | 52.9 / 50.6 (85%) | 55.7 / 52.2 (81%) | 61.4 / 59.4 (74%) | 64.1 / 100.7 (71%) | 70.8 / 104.6 (65%) |
| 8448x7168 | layer-0 gate_up | 20.8 / 23.8 (92%) | 21.7 / 25.4 (88%) | 24.4 / 28.1 (79%) | 26.3 / 29.4 (74%) | 28.0 / 35.1 (70%) | 30.7 / 35.9 (64%) | 33.7 / 39.6 (58%) | 37.0 / 43.7 (54%) | 41.1 / 42.6 (49%) |
| 7168x4224 | layer-0 down | 10.7 / 12.4 (97%) | 12.2 / 13.4 (85%) | 13.4 / 14.9 (78%) | 14.7 / 16.1 (72%) | 15.8 / 17.6 (67%) | 18.8 / 20.0 (57%) | 20.6 / 22.7 (52%) | 23.6 / 24.2 (46%) | 26.3 / 23.8 (42%) |
| 2880x7168 | drafter context_kv | 9.2 / 12.4 (82%) | 9.6 / 15.7 (80%) | 10.8 / 20.5 (71%) | 12.5 / 16.8 (62%) | 13.3 / 23.6 (59%) | 14.5 / 19.7 (54%) | 16.2 / 21.2 (49%) | 18.7 / 24.2 (43%) | 22.0 / 25.7 (37%) |
| 7168x35840 | drafter context_proj | 79.2 / 89.5 (96%) | 78.4 / 89.4 (97%) | 81.7 / 91.8 (94%) | 85.2 / 110.5 (90%) | 88.5 / 124.0 (87%) | 92.7 / 127.5 (83%) | 101.2 / 141.5 (77%) | 115.5 / 287.7 (68%) | 123.1 / 277.3 (64%) |

## Tests
- `python3 -m pytest tests/gemm/bf16_skinny_common -q` (every shape x 21 row counts against the fp64 reference,
  bitwise repeats, NaN / inf, graph replays after the warm-up with no compile, XCD placement)
- `python3 benchmarks/gemm/bf16_skinny_common/benchmark_linear.py [--shapes N,K ...] [--rows M ...]`
