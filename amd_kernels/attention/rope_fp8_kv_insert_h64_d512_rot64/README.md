# Query RoPE + e4m3 quant and KV RoPE + e4m3 cache insert, 64 heads of 512, rotary 64

## What it does

One launch turns a layer's projected query and KV row into the sparse attention's inputs: the bf16 query `[rows, 64,
512]` becomes e4m3 (GPT-J RoPE on dims 448..511, times the query scale inverse, saturating at +-448), and the bf16 KV
row `[rows, 512]`, rotated the same way and divided by the KV scale, is inserted as e4m3 into the sliding-window cache
`[pages, block, 512]` at its slot (slot < 0 skips; the queries are never skipped). Position `p` rotates by row `p` of
the fp32 `cos_sin_cache [rows, 64]` (cos | sin). The queries are not normed here (the caller norms them before the
projection). Served: 64 query heads, 1 KV head, block 128, unit scales, 40 layers. Contract: `reference.py`. Values
stay fp32 from the bf16 input to the e4m3 conversion (no bf16 rounding in between), with the RoPE as fma(e, cos, -(o *
sin)) and fma(o, cos, e * sin); the result is bit for bit the C op of the serving stack that it replaces (the test
compares both).

## Entry points

- `rope_fp8_kv_insert`: fill the caller's e4m3 query tensor and insert the KV rows into the cache.
- `supports_rope_fp8_kv_insert`: whether the inputs fit (gfx950, bf16 query, fp32 cos | sin table).

An entry (`amd_kernels.warm_up`): the first call per page size compiles its kernel. The row count is a runtime
argument: no dispatch.

## Performance

One MI355X; one graph of up to 100 launches with their own inputs (56 at 768 rows: 4 GiB), A B B A, median of 5 x 10
replays, us per call. The baseline is the vLLM C op `fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert`. The
kernel moves 4.7 TB/s at 768 rows.

| Rows | Kernel | Baseline | Speedup |
|---|---|---|---|
| 16 | 2.32 | 3.85 | 1.66x |
| 48 | 2.91 | 4.65 | 1.59x |
| 96 | 4.03 | 5.98 | 1.48x |
| 192 | 6.20 | 8.27 | 1.33x |
| 384 | 9.67 | 13.05 | 1.35x |
| 768 | 16.43 | 22.35 | 1.36x |

## Tests

```bash
python3 -m pytest tests/attention/rope_fp8_kv_insert_h64_d512_rot64 -q
python3 benchmarks/attention/rope_fp8_kv_insert_h64_d512_rot64/benchmark_rope_fp8_kv_insert.py
```
