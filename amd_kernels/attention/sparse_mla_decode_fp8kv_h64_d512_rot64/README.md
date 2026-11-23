# Sparse MLA decode, 64 heads x d512, e4m3 KV, inverse-RoPE MXFP8 output

## What it does
Decode attention of DeepSeek-V4.1-Flash at DEP8 (64 query heads, one KV head, head dim 512 = 448 nope + 64 RoPE dims):
each query row attends a ragged main list (SWA, <= 128 keys) followed by an extra list (top-512 compressed keys) as one
key set, with the attention sink as a key without a value. Q and KV are per-tensor e4m3
(`rope_fp8_kv_insert_h64_d512_rot64` writes both); the scales enter as `q_scale * kv_scale * softmax_scale`. The
epilogue rounds O / sum to bf16, applies the inverse GPT-J RoPE to dims 448..511 at `positions`, and quantizes MXFP8:
e4m3 codes `[T, 8 * 4096]` + row-major uint8 E8M0 scales `[T, 8 * 128]` (1 x 32 blocks), exactly the activation of
`gemm/mxfp8_grouped_g8_n1024_k4096` (`grouped_gemm`): no cast / quant launch between the two. `out_bf16=` (no RoPE) is
the test mode. A row with no key gets zeros. <= 1024 keys per row, 16..768 rows (mixed 1..6 per request), block sizes
32/64/128, padded page pitch. Deterministic (no atomics into outputs), graph-safe. `reference.py`: fp64 attention, the
inverse RoPE and `quantize_mxfp8` reference; cos >= 0.99999 (P as two e4m3 terms), exact MXFP8 bytes given the same
bf16. Each row's heads split over `64 / hg` workgroups (`hg` = 16, 32, 64 by row count: 64 rows or fewer, 128 or fewer,
more), so a few rows use many CUs and no split-KV combine runs; the 64-head kernel is persistent (a workgroup runs
several rows in a row, the next row's first tiles load during the epilogue). Large scores (q codes saturated at 448,
scales of 8-1000, softmax scores of hundreds of nats) give finite outputs with no NaN code or 255 scale, within 3%
relative L2 error per head of the fp64 reference (the MXFP8 quantization; AITER's `sparse_mla_fwd` up to 7%).

## Entry points
- `sparse_mla_decode` (`@warm_up.entry`): the attention + epilogue; `p_terms=1` (one e4m3 P, cosine ~0.9997 like the
  baseline, 10-15% faster); `hg=`, `wgs=` are test knobs; `splits=S` is a key-range split with a fixed-order combine
  (slower).
- `supports_sparse_mla_decode`: whether the inputs fit (gfx950, 64 heads, e4m3, 512-byte rows).

Dispatch: `host.plan_hg` (head group by row count) and `host.plan_wgs` (persistent grid).

## Performance
One MI355X (shared); graph replay of 100 distinct launches (window 128 + top-512, 80% of the list shared inside a
request), A B B A, two passes, us; baseline AITER Triton `sparse_mla_fwd` + cast / quant.

| T | AITER us | ours us | x | p_terms=1 us | x |
|---|---|---|---|---|---|
| 16 | 17.2 | 13.8 | 1.25 | 12.8 | 1.35 |
| 32 | 18.7 | 14.3 | 1.31 | 13.3 | 1.41 |
| 48 | 18.9 | 14.5 | 1.31 | 13.4 | 1.41 |
| 64 | 20.7 | 14.6 | 1.42 | 13.5 | 1.53 |
| 96 | 21.1 | 15.4 | 1.37 | 14.9 | 1.41 |
| 128 | 21.5 | 15.7 | 1.37 | 15.2 | 1.42 |
| 192 | 27.3 | 22.6 | 1.21 | 20.8 | 1.31 |
| 256 | 28.3 | 24.1 | 1.17 | 21.6 | 1.31 |
| 384 | 45.5 | 41.2 | 1.10 | 37.4 | 1.22 |
| 512 | 53.6 | 45.0 | 1.19 | 40.4 | 1.33 |
| 768 | 77.9 | 66.7 | 1.17 | 58.0 | 1.34 |

At T = 768 the key-tile loop runs at the measured L2 / MALL -> LDS row-gather ceiling (5.57 TB/s, 1.45 us per 32 KB
tile); the whole kernel reaches 68% of it (3.8 TB/s) because every row pays its prologue and epilogue (FP8 MFMA 24-36%,
HBM 31%: neither binds).

## Tests
```bash
python3 -m pytest tests/attention/sparse_mla_decode_fp8kv_h64_d512_rot64 -q
python3 benchmarks/attention/sparse_mla_decode_fp8kv_h64_d512_rot64/benchmark_decode.py
```
Served shapes T=16..768, edge key counts, blocks, pitch, SWA-only, peaked / flat scores, padding rows, the three
head-group kernels and the persistent one (`hg`, `wgs`) against `reference.py`, bitwise repeats, graph replay into
NaN-filled outputs, splits, one-term P, the rope -> attention -> wo_a chain, the warm-up. `test_saturation.py`:
saturated q (1-100%), scales 8 / 1000, masked blocks, single key, score jumps between tiles; every kernel, split path.
