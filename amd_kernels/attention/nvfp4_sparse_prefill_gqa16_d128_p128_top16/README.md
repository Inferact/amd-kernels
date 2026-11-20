# NVFP4 sparse prefill attention, GQA16, D128, 128-token pages, top-16

## What it does

MiniMax-M3.1 sparse attention: each query attends its selected 16 blocks of 128 keys (own block causal).
Inputs: Q `[T, 16 Hkv, 128]` (bf16, fp16 for fp16 compute, or prepacked e4m3 u8 for fp8), NVFP4 pages
`[pages, 2 Hkv, 128, 72]` u8 with free page stride, logical picks `[Hkv, T, 16]` i32 (-1 none), block tables, request
ids and positions. Tiles may span requests. Output: bf16, MXFP8 codes + AITER-shuffled scales for o_proj, or
sequence-parallel all-to-all rows (Hkv 1); the fused outputs leave the bf16 output and the padding rows untouched.
Served: Hkv 1 (TP4) and Hkv 2 (TP2), up to 65536 tokens, fp16 with the lazy rescale (fp16 P against a reference
re-based only past 2^8, P <= 256; the fastest at every T and both Hkv); bf16, plain fp16 and fp8 stay available. FP8
uses NV Q8KV4 with P rounded per 32-key chunk; gate it with the NV KL check.
Contract: `reference.py` (`REL_L2_TOL`, `COS_MIN`); fused outputs equal quantization of the kernel's own bf16 output.
Strict xfail: bf16 TQ4 can propagate a NaN V scale past the sequence end through zero P.

## Entry points

- `nvfp4_sparse_prefill`: the attention of a prefill step into bf16, or fused into the MXFP8 o_proj input or the
  sequence-parallel all-to-all rows; graph-safe. An `amd_kernels.warm_up` entry: the first call of each mode warms
  every variant at 8 / 512 / 1024 / 1536 / 2048 tokens (Hkv 2: 8 / 256 / 384 / 1024).
- `nvfp4_sparse_prefill_bf16`, `nvfp4_sparse_prefill_fp16`, `nvfp4_sparse_prefill_fp8`: the same with the compute
  precision bound. Build the fp8 table outside graph capture.
- `host.warm_up_prefill`: compiles every bf16-output variant of one compute precision now.
- `supports_nvfp4_sparse_prefill`: whether a model fits (gfx950, GQA16, D128, top-16 blocks of 128).

`host.py` `nvfp4_sparse_prefill` picks the kernel by precision, KV heads, token count and grid size (`_local_tile`).

## Performance

MI355X, fp16 lazy (served), bf16 Q and output (what vLLM calls), 16K prefix, 100 cold inputs, 2 alternated
graph-replay rounds, us. Hkv 2 baseline: no small tiles, the four-query kernel at every count (one round below 769
tokens; the served kernel from 1024).

| Tokens | 128 | 256 | 512 | 768 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|---|---|
| Hkv 1 (TP4) | 14.1 | 14.5 | 21.0 | 29.1 | 29.7 | 47.0 | 87.4 | 167.4 | 329.5 | 658.3 |
| Hkv 2 (TP2) | 14.5 | 20.7 | 29.8 | 39.8 | 47.5 | 87.7 | 170.2 | 328.7 | 654.8 | 1315.1 |
| Hkv 2, baseline | 38.5 | 38.1 | 38.1 | 45.9 | 47.9 | 88.9 | 170.5 | 335.1 | 667.3 | - |

## Tests

```bash
python3 -m pytest tests/attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16 -q
python3 benchmarks/attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16/benchmark_sparse_prefill.py --fp8-bf16q --fp16-lazy
python3 benchmarks/attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16/benchmark_sparse_prefill.py --fp16-lazy --only 2:128,2:256,2:512,2:768,2:2048,2:8192
```
