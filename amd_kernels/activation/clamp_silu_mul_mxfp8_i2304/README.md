# Clamped SiLU-mul + MXFP8, I 2304 (gfx950)

## What it does

One launch for the activation between the shared-expert gate_up and down GEMMs of a SwiGLU MLP with a clamp, and the
MXFP8 quant of its result for the down GEMM. Input: gate_up bf16 `[M, 4608]` (gate columns 0-2303, up columns 2304-4607,
unit column stride, any row stride), M >= 0 (decode rows 1-768, any larger M works), fp32 limit (10 in the served
model). Outputs: e4m3 codes `[M, 2304]` and E8M0 scale bytes `[M, 72]`, both row-major (the layout the vLLM MXFP8 linear
takes as a pre-quantized `QuantizedActivation`).

Contract (`reference.py`): vLLM's `SiluAndMulWithClamp` at alpha 1, beta 0 (its HIP op): `g = min(gate, limit)`,
`u = clamp(up, -limit, limit) + 0`, `act = bf16(bf16(silu(g)) * u)` with `silu(g) = g / (1 + exp(-g))` in fp32 (the
intermediate bf16 rounding is the op's); then Triton's `mxfp8_e4m3_quantize` of the bf16 `act`: per 32 columns
`clamp(ceil(log2(amax / 448)) + 127, 0, 254)` and e4m3 codes of `act * 2^(127 - byte)`. Codes and scales equal the
served op + quantizer bit for bit (tests, 19 row counts).

## Entry points

- `clamp_silu_mul_mxfp8`: the down GEMM's codes and scales of a gate_up block; outputs may be the caller's static
  buffers; graph-capture safe, bitwise repeatable.
- `warmup_clamp_silu_mul_mxfp8`: compiles the one variant before graph capture (also the first call's warm-up).

The kernel has one variant at every M (`kernel.py`: 576 lanes per row, four columns each).

## Performance

One MI355X, graph replay of 100 distinct inputs, median of 5 x 5 replays, rounds A B B A, us per call. Baseline: vLLM's
`silu_and_mul_with_clamp` HIP op + Triton `mxfp8_e4m3_quantize` (two launches).

| rows | 16 | 32 | 64 | 96 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|---|
| served op + quant | 4.56 | 4.82 | 5.06 | 5.06 | 5.10 | 5.22 | 5.53 | 6.46 | 7.00 | 8.02 |
| this entry | 2.23 | 2.38 | 2.47 | 2.53 | 2.55 | 2.67 | 2.89 | 3.43 | 3.61 | 4.58 |

## Tests

```bash
python3 -m pytest tests/activation/clamp_silu_mul_mxfp8_i2304 -q
python3 benchmarks/activation/clamp_silu_mul_mxfp8_i2304/benchmark_clamp_silu_mul.py
```

One GPU: every decode row count against the reference and the served ops (bitwise), clamp and sign-of-zero edge values,
row strides, caller buffers, graph replay with poisoned outputs, determinism, no compile after the warm-up.
