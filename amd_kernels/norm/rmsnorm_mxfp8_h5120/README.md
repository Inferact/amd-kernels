# RMSNorm + MXFP8 quant, H 5120 (gfx950)

## What it does

The RMSNorm of a hidden-size row that feeds MXFP8 GEMMs, in one launch: bf16 `[M, 5120]` in (unit column stride, any row
stride), bf16 weight `[5120]`, fp32 eps (1e-20 in the served model); out: e4m3 codes `[M, 5120]` and E8M0 scale bytes
`[M, 160]`, both row-major, which is the data and scale of vLLM's pre-quantized `QuantizedActivation` for the MXFP8 linear.
M >= 0 (draft context rows 1-1024, prefill chunks of 8192).

Contract (`reference.py`): `y = bf16(x * rsqrt(mean(x^2) + eps) * w)` in fp32 with one bf16 rounding, quantized from
that bf16 value as vLLM's `mxfp8_e4m3_quantize`: per 32 columns the byte `clamp(ceil(log2(amax / 448)) + 127, 0, 254)`
and e4m3 codes of `y * 2^(127 - byte)`. The square sum is the kernel's own fixed tree and the rsqrt the hardware's, so
about 1e-5 of the bf16 values of `y` differ from torch's by an ulp (2e-6 of the codes; the tests bound it, check the
dequantized rows' cosine >= 0.99999 and that two runs match bit for bit).

Producer and consumer: the draft's `main_norm` takes the output of
[`gemm/mxfp8_n5120_k15360`](../../gemm/mxfp8_n5120_k15360/README.md) (`main_proj`, bf16 `[M, 5120]`) and its codes and
scales feed each draft layer's `wq_a | wkv` GEMM as a pre-quantized activation (`gemm.mxfp8_common.gemm.mxfp8_gemm`), so
no quantizer launch sits between the norm and the GEMMs; one result serves all three layers.

## Entry points

- `rmsnorm_mxfp8`: codes and scales of a row block; outputs may be the caller's static buffers; graph-capture safe, no
  atomics, bitwise repeatable.
- `warmup_rmsnorm_mxfp8`: compiles the one variant before graph capture (also the first call's warm-up).

One variant at every M (`kernel.py`: a workgroup of 320 lanes per row, 16 columns per lane).

## Performance

One MI355X, graph replay of up to 100 distinct inputs, median of 5 x 5 replays, rounds A B B A, us per call. Baseline:
the served pair, vLLM's `rms_norm` op (bf16 out) and its Triton `mxfp8_e4m3_quantize` (two launches).

| rows | 16 | 40 | 80 | 160 | 192 | 320 | 384 | 640 | 768 | 1024 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rms_norm + quantize | 4.58 | 5.05 | 5.14 | 5.51 | 5.78 | 8.05 | 8.62 | 10.40 | 11.15 | 12.51 | 29.44 | 54.41 |
| this entry | 2.21 | 2.43 | 2.51 | 2.74 | 2.93 | 3.79 | 3.93 | 4.74 | 5.03 | 5.72 | 14.31 | 25.47 |

## Tests

```bash
python3 -m pytest tests/norm/rmsnorm_mxfp8_h5120 -q
python3 benchmarks/norm/rmsnorm_mxfp8_h5120/benchmark_rmsnorm.py
```

One GPU: 20 row counts (1-4097) against the reference, extreme rows (zero, overflow, underflowing squares, one hot
column), row strides, caller buffers, graph replay with poisoned outputs, determinism, no compile after the warm-up.
