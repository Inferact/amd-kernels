# MoE router, E 896 x H 7168, sigmoid + correction bias + top-16

## What it does
One launch: `topk_weights fp32 [M, 16]`, `topk_ids int32 [M, 16]` from `x bf16 [M, 7168]`, the gate `W bf16 [896, 7168]`
(replicated) and the fp32 `e_score_correction_bias [896]`, M <= 192 (above that, launches of 192 rows): the bf16 GEMM
(fp32 sums), AITER's `biased_grouped_topk` with one group (sigmoid, + bias, top-16, renormalise, times
`routed_scaling_factor`) and, optionally (`aux`), the BM=16 MoE sort of those 16 experts in AITER's auxiliary layout
(`register_presorted_moe`: sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, m_indices,
reverse_sorted) and the zeroed [M, 3584] bf16 buffer AITER's atomic stage 2 accumulates into.

Contract (`reference.py`): the top-16 is bit for bit `aiter.biased_grouped_topk` (the `topk_reg_kernel` serving runs) on
the logits the kernel sums (bf16 products in fp32, split-K slabs added in slice order, each slab word with its lowest
mantissa bit cleared: <= 1 ulp per slab; `router_logits` returns exactly these): ids, slot order (AITER's tie rank for
equal choices) and weights `sigmoid * (scale / sum)`. The sort is deterministic (an expert's rows in ascending token
order; padding rows: token id M, weight 0, m_indices M) and bit for bit `moe_sort_reference`. Two runs give the same bits.
W layouts: `row` (the checkpoint's) or `shuf` (`shuffle_weight`, a bit-exact permutation made once at load). One call
at a time per stream (the workspace holds the epoch words); CUDA-graph safe, no atomics into outputs.

## Entry points
- `router_sigmoid_bias_top16_e896_h7168`: the router (an `amd_kernels.warm_up` entry: the first call of each mode
  compiles every row-tile variant).
- `make_aux`: the sort's output tensors of one M (AITER's capacity).
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout (replace the parameter once).
- `router_logits`, `top16`: the GEMM half and the top-16 launch alone, for tests and benchmarks; `supports`: whether
  the weight and bias fit.
The dispatch (16-row tile -> K splits, column waves, row splits) is `CONFIG_BY_MT` in `host.py`; the hand-off between
the GEMM and the top-16 is in the module docstring of `kernel.py`.

For prefill, `prefill_top16_mxfp4(logits, latent, bias, routed_scaling_factor, valid_rows)` consumes existing projections, up to 4096 rows. One launch selects routes and packs the 3584-column latent to MXFP4. FP32 latent rounds to BF16 before quantization. Both independent projections and aligned slices of a combined projection are supported. Padding has zero weights and IDs of -1. The quant output includes every input row. Call once with each latent dtype before graph capture.

## Performance
MI355X, graph, cold W (24 copies), min of 15, us (`benchmark_router.py`; launch floor 1.73). stock = hipBLASLt mm + AITER.
`skel` = the same kernel with no MFMAs and no top-16 (the measured critical path of the hand-off), `stream` = the weight
stream alone; %skel = skel / fused, %HBM = (12.8 MB at 7.0 TB/s + launch) / fused.

| M | cfg | stock | fused | +aux | speedup | stream | skel | %skel | %HBM |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 16:4:1 | 16.7 | 6.8 | 9.0 | 2.45x | 4.4 | 5.7 | 83 | 52 |
| 12 | 16:4:1 | 15.5 | 7.6 | 10.2 | 2.04x | 4.4 | 6.0 | 78 | 47 |
| 24 | 16:4:1 | 15.4 | 8.0 | 11.0 | 1.93x | 4.4 | 6.5 | 81 | 45 |
| 40 | 16:4:1 | 15.4 | 8.5 | 12.1 | 1.81x | 4.4 | 7.1 | 83 | 43 |
| 65 | 8:4:1:2 | 17.2 | 9.4 | 13.3 | 1.82x | 4.7 | 7.8 | 82 | 39 |
| 80 | 8:4:1:2 | 19.4 | 9.5 | 13.5 | 2.05x | 4.7 | 8.1 | 85 | 39 |
| 128 | 8:4:1:2 | 19.3 | 11.0 | 14.6 | 1.75x | 4.7 | 9.3 | 84 | 35 |
| 192 | 8:4:1:2 | 21.4 | 12.6 | 16.8 | 1.70x | 4.7 | 10.6 | 84 | 31 |

The fused call runs at 78-85% of `skel`, which still lacks the ~1.6 us top-16 chain: the tail after the last slab lands
(store visibility + poll ~1.3 us, token chain ~2.0 us) and, from M=65, the L2 -> LDS bytes of the tiles (x is re-read by
every column tile) remain.

## Tests
```bash
python3 -m pytest tests/moe/router_sigmoid_bias_top16_e896_h7168 -q
python3 benchmarks/moe/router_sigmoid_bias_top16_e896_h7168/benchmark_router.py [M ...]
python3 benchmarks/moe/router_sigmoid_bias_top16_e896_h7168/benchmark_prefill.py [--packed]
```
