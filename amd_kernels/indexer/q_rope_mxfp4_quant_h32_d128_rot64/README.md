# Indexer query RoPE + MXFP4 quant, 32 heads of 128, rotary 64

## What it does

One launch turns the indexer's bf16 query `[rows, 32, 128]` into what the MQA-logits kernel reads: GPT-J RoPE on dims
64..127 (pair `(2j, 2j + 1)` by `cos_sin_cache[position]`, fp32 or bf16 `[rows, 64]`, the rotated values rounded to
bf16), then MXFP4 per 32 dims (e8m0 scale `2^ceil(log2(amax / 6))` from the exponent bits, e2m1 with ties to even, two
values a byte, the even dim in the low nibble). Outputs: packed `u8 [rows, 32, 64]`, scales `u8 [rows, 32, 4]` (what the
caller views as one int32 a head) and `fp32 [rows, 32] = bf16 weights * weights_scale`. Contract: `reference.py`;
bit for bit AITER's `q_rope_mxfp4_quant` (the test compares both). Served: 8 layers.

## Entry points

- `q_rope_mxfp4_quant`: quantize the query and scale the weights (pass the outputs under graph capture).
- `supports_q_rope_mxfp4_quant`: whether the inputs fit (gfx950, contiguous bf16 query, fp32 / bf16 table).

An entry (`amd_kernels.warm_up`): the first call per table dtype compiles its kernel. The row count is a runtime
argument: no dispatch.

## Performance

One MI355X; one graph of 100 launches with their own inputs, A B B A, median of 5 x 10 replays, us per call. The
baseline is AITER's Triton op: a tie below 384 rows (both sit on the launch floor of 1.8 us), 5-12% faster above.

| Rows | Kernel | Baseline | Speedup |
|---|---|---|---|
| 16 | 1.88 | 1.80 | 0.96x |
| 48 | 2.23 | 2.16 | 0.97x |
| 96 | 2.36 | 2.31 | 0.98x |
| 192 | 2.52 | 2.43 | 0.96x |
| 384 | 3.01 | 3.38 | 1.12x |
| 768 | 4.10 | 4.30 | 1.05x |

Limit: launch (graph node 1.59 us) at 16 rows, the cold read of the operands at 192 / 768 rows (2.3 / 3.8 us); achieved
85 / 91 / 92% at 16 / 192 / 768 rows. Beating the baseline at fewer rows needs the launch removed (the quant in the GEMM
epilogue that produces the query).

## Tests

```bash
python3 -m pytest tests/indexer/q_rope_mxfp4_quant_h32_d128_rot64 -q
python3 benchmarks/indexer/q_rope_mxfp4_quant_h32_d128_rot64/benchmark_q_rope_mxfp4_quant.py
```
