# NVFP4 sparse decode attention, 16 query heads per KV head, head dim 128, 128-token pages, top-16

## What it does

MiniMax-M3.1's sparse decode attention over the NVFP4 K / V cache: each query row attends over the 16 128-token
blocks the indexer picked (`topk_idx [Hkv, rows, topk]` int32 logical blocks, -1 none, the own block causal), the
pages `[pages, 2 Hkv, 128, 72]` u8 (`reference.py`; free page stride) dequantized in registers; bf16 `[rows, H, 128]`
or the o_proj's MXFP8 input out. Served: TP4 1 KV / 16 q heads (1-256 requests), TP2 2 KV / 32 q heads (1-128
requests), 1 / 4 / 8 rows per request, compute bf16, fp16 (fastest, below) or fp8. Contract `sparse_decode_reference`
(fp64, `REL_L2` 5e-3, `COS` 0.99999, splits in split order): bf16 K / V `e4m3_rn_satfinite(e2m1 x scale)` (NV's; rel
L2 to exact K / V 3.1-3.3e-2), fp16 exact (1.6-1.7e-3), fp8 NV Q8KV4, P per 32-key chunk; MXFP8 = vLLM's quantizer
bytes.

## Entry points

- `nvfp4_sparse_decode`: the attention of a step's decode rows (row i is row i % ql of request i // ql) into bf16 or
  the o_proj's MXFP8 input (row-major or in the decode GEMM's 16-row tiles); given the indexer's scores it also
  writes `topk_idx` (bitwise `nvfp4_index_score_top16`, selected inside the attention at 1 / 2 KV heads up to 256 /
  1024 rows, index head h = KV head h, bf16 / fp16). An `amd_kernels.warm_up` entry: a mode's first call compiles
  1-256 requests. Graph-safe.
- `host.warm_up_decode`: compiles both outputs at init (else a mixed step's bf16-output mode compiles late).
- `index_decode.nvfp4_index_sparse_decode`: a sparse layer's decode block in one call (index top-16, then the
  attention); returns its route, bitwise equal on every route: `split4` (four GPUs), `split2` (two GPUs), `fused` (one
  GPU, bf16 / fp16), `chain`. `index_decode.warm_up` compiles it.

`host.py` picks the variant, splits and grid by rows and KV heads (`_tuned_splits`, `decode_splits`, the tail grid);
`index_decode.route` the route by request count.

## Performance

Graph replay, 100 calls with own pages (K / V cold), 60K-100K, 25 distinct picks per 4 rows, us per call, MXFP8 out,
one process, two rounds (fp16 plain: fp16 without its small-batch schedule; TP2 untuned: the 2-KV-head launch without
its schedule, the general kernel); bf16 out 0.2-0.6 us less. TP2 at N requests = TP4 at 2N (the same (row, KV head)
pairs); vs AITER's Gluon sparse PA: 0.55-0.73x the time:
| ql x requests | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|---|
| TP4 4: bf16 / fp16 plain / fp16 / fp8 | 8.46 / 10.35 / 8.15 / 8.78 | 8.90 / 10.59 / 8.59 / 9.28 | 9.55 / 11.43 / 9.24 / 9.87 | 10.19 / 9.82 / 9.85 / 10.52 | 11.03 / 10.39 / 10.42 / 11.46 | 13.57 / 12.45 / 12.44 / 13.85 | 18.18 / 16.58 / 16.58 / 18.55 | 29.03 / 26.92 / 26.97 / 28.49 |
| TP4 8: bf16 / fp16 plain / fp16 / fp8 | 8.35 / 10.62 / 8.20 / 8.46 | 8.72 / 11.44 / 8.55 / 8.84 | 9.29 / 12.35 / 9.19 / 9.38 | 10.39 / 10.33 / 10.35 / 10.48 | 12.63 / 12.17 / 12.15 / 12.49 | 17.04 / 15.82 / 15.81 / 16.84 | 28.01 / 25.58 / 25.60 / 26.52 | |
| TP2 4: fp16 untuned / fp16 | 10.63 / 8.21 | 11.41 / 8.56 | 12.59 / 9.49 | 10.42 / 10.38 | 12.44 / 12.32 | 16.89 / 16.74 | 27.74 / 27.25 | 47.18 / 47.34 |
| TP2 8: fp16 untuned / fp16 | 11.42 / 8.50 | 12.34 / 9.12 | 12.19 / 10.60 | 12.19 / 12.13 | 15.89 / 15.82 | 25.62 / 25.84 | 44.88 / 43.39 | 80.62 / 81.17 |

TP2 ql 8 between the whole CU rounds (the tail grid): 24 / 40 / 52 requests untuned 23.25 / 34.13 / 44.17, fp16
22.22 / 31.33 / 39.18 us. bf16 out, fp16 vs bf16: ql 8 x 8 +0.00..+0.08 us (three runs), elsewhere faster; with
`index_score`: -0.04..-1.90 us MXFP8 out, bf16 out ql 4 x 1-8 within +0.08 us.

## Tests

bf16 / fp16 / fp8, 1 and 2 KV heads; strict xfails: mixed steps without `warm_up_decode`, the MXFP8 subnormal scale
byte.
```bash
python3 -m pytest tests/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16 -q   # *_ranks4: 4 GPUs, *_ranks2: 2
python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_sparse_decode.py
python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_oproj_input.py --hkv 2
python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode.py --hkv 2
python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode_ranks2.py --ql 8  # 2 GPUs
```
