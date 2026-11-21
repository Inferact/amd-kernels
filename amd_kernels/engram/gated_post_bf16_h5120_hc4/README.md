# Gated Engram update, bf16, hidden 5120, 4 residual copies

## What it does
vLLM's Engram gate and residual update (`_fused_engram_post_wkv_kernel`), for the layers that read an n-gram
memory: `hidden` bf16 `[T, 4, 5120]` (the four mHC streams), `kv` bf16 `[T, 25600]` (the `wkv` output: four keys of
5120, then the value shared by the copies), `q_weight` / `k_weight` bf16 `[4, 5120]`, optional bool `token_mask` `[T]`
(False: the token passes through). Per (token, copy), in fp32: `dot = sum(h * q * k * key) * rsqrt(mean(h^2) + eps) *
rsqrt(mean(key^2) + eps) / sqrt(5120)`, `gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))`, output `h + gate * value`
rounded once to bf16 (eps 1e-20). Contract and tolerance (`reference.py`): cosine >= 0.99999, within a bf16 ulp of the
larger term, under 1e-3 of the elements differ from the torch reference (the kernel's sums associate differently, so
it is not bitwise; it is closer to the exact value than torch's, and matches vLLM's Triton kernel the same way).
Repeat runs are bit-identical; the output may alias `hidden`.

## Entry points
- `gated_post`: the update for a decode batch or a prefill chunk, eager or in a CUDA graph; `out=hidden` updates in
  place. The first call of each mode (mask or none) warms up (`amd_kernels.warm_up`); the token count is a runtime
  argument. `gated_post_reference` is the contract.

One kernel per mask mode; `host.py` fixes the 5-vectors-per-lane layout (measured against 1, 2 and 10).

## Performance
MI355X, graph replay over 100 distinct (hidden, kv) pairs (far above the LLC), median of 5 rounds x 5 replays, us per
call, token mask, separate output; baseline vLLM's Triton kernel (8 warps). One GPU shared with other jobs: +-3%.

| rows | 16 | 32 | 64 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|
| this entry | 3.4 | 4.2 | 4.6 | 6.0 | 7.0 | 8.1 | 10.8 | 15.0 | 19.9 |
| vLLM Triton | 3.3 | 3.7 | 4.5 | 5.9 | 7.4 | 8.9 | 12.2 | 15.1 | 21.1 |

At 768 rows the kernel moves 102 MB (hidden read, kv read, output write) in 19.9 us: 5.1 TB/s, 64% of the 8 TB/s
HBM peak. Below 128 rows it is launch-bound and equal to Triton: the entry is a parity port, its value is the
`token_mask` / in-place contract and one fewer dependency, not speed.

## Tests
```bash
python3 -m pytest tests/engram/gated_post_bf16_h5120_hc4 -q               # also checks vLLM's Triton kernel if installed
python3 benchmarks/engram/gated_post_bf16_h5120_hc4/benchmark_gated_post.py
```
