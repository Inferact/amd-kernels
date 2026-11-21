# MXFP4 routed experts, H3584 I384 E896 top16 (gfx950)

## What it does

The 896 routed experts of one TP8 rank of Kimi-K3.1 (moe_intermediate 3072 / 8 = 384 per rank, latent hidden 3584,
top-16, all experts on every rank). Input: the MXFP4 activations (e2m1 `[M, 1792]` + e8m0 `[M, 112]` per 32, from
`quantize_activations` or the latent down-projection's epilogue) and the router's BM=16 sort. Per route: gate/up GEMM1
(SiTUv2 `beta 4`, `linear_beta 25`, no clamp; the activation re-quantized to MXFP4 with AITER's rule) and down GEMM2,
the router weight applied once; a token's output = its 16 slot rows summed in slot order in fp32, rounded once to
BF16. Weights are read in vLLM's AITER layouts (`shuffle_weight`, `e8m0_shuffle`), nothing is re-laid-out. Contract
(`reference.py`): quant bit for bit, `moe_ref` within rel. L2 2e-2 (observed 2.7e-3; AITER's own is 5.5e-3 from bf16
atomic adds); no atomics into outputs, two runs bit for bit, graph replay = eager. M 1-192.

## Entry points

- `MoeWeights.from_shuffled` / `MoeWeights.from_checkpoint`: a layer's experts as vLLM holds them / from row-major
  checkpoint tensors (shuffled once), before graph capture.
- `mxfp4_moe_experts`: the experts of one call, GEMM1 + GEMM2 in one launch then the slot sum (an
  `amd_kernels.warm_up` entry: the first call compiles everything).
- `Sort.from_aux`: the sort from the router entry's `make_aux` (`moe/router_sigmoid_bias_top16_e896_h7168`).
- `quantize_activations`, `moe_sort`, `mxfp4_moe`: bf16 activation quant, AITER's sort, bf16 in -> out (tests and
  benchmarks).
No row-count dispatch: one kernel set serves M 1-192 (`host.py`).

## Performance

One MI355X shared with other jobs (quiet-GPU times are lower), 3 layers of weights (5.9 GB), CUDA graph replay,
interleaved min of 25 per layer, us. Ceiling: the
measured nontemporal HBM read, 7.0 TB/s (6.4-6.7 for plain loads). TB/s = unique experts x 2.19 MB + the activations,
slots and output. Stock fused_moe = AITER with the shipped tuned rows incl. its own sort and quant (what vLLM runs);
stock stages = AITER's kernels on patched BM16 rows with the same sort (`register_presorted_moe`). Ours = quantize +
experts + slot sum, the BM16 sort handed in.

| M | routing | unique | us | TB/s | % of 7.0 | vs stock fused_moe | vs stock stages |
|---|---|---|---|---|---|---|---|
| 12 | uniform | 175 | 66.1 | 5.89 | 84% | 1.20x | 1.05x |
| 12 | zipf1.0 | 121 | 49.2 | 5.48 | 78% | 1.22x | 1.01x |
| 24 | uniform | 321 | 110.4 | 6.47 | 92% | 1.03x | 1.01x |
| 24 | zipf1.0 | 194 | 72.8 | 5.96 | 85% | 1.09x | 1.06x |
| 40 | uniform | 474 | 158.2 | 6.68 | 95% | 1.03x | 1.01x |
| 40 | zipf1.0 | 282 | 100.8 | 6.27 | 90% | 1.06x | 1.04x |
| 65 | uniform | 613 | 203.5 | 6.73 | 96% | 1.06x | 1.02x |
| 65 | zipf1.0 | 360 | 128.5 | 6.31 | 90% | 1.08x | 1.00x |
| 80 | uniform | 676 | 226.6 | 6.67 | 95% | 1.03x | 1.02x |
| 80 | zipf1.0 | 422 | 148.2 | 6.42 | 92% | 1.06x | 1.01x |
| 128 | uniform | 812 | 278.4 | 6.55 | 94% | 1.03x | 1.01x |
| 128 | zipf1.0 | 535 | 189.9 | 6.39 | 91% | 1.01x | 1.00x |
| 192 | uniform | 866 | 298.0 | 6.57 | 94% | 1.00x | 1.05x |
| 192 | zipf1.0 | 650 | 241.4 | 6.14 | 88% | 0.98x | 1.02x |

rocprofv3 FETCH_SIZE (best of 12 replays): the one-launch kernel fetches 6.5 TB/s at M 12, 6.9 at 24, 7.0-7.1 at
M 40-192 (MFMA busy 2.6-4%). Below the ceiling at M 12-24: sparse routing is launch / ramp bound (3 kernels, ~3 us each).

## Tests

```bash
python3 -m pytest tests/moe/mxfp4_h3584_i3072_e896_top16 -q   # reference, AITER, determinism, fused = unfused, warm-up
python3 benchmarks/moe/mxfp4_h3584_i3072_e896_top16/benchmark_experts.py [M ...]
```
