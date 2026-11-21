# Dual RMSNorm + MXFP8, q 1280 / kv 512 (gfx950)

## What it does

One launch for the two RMSNorms that follow the fused q-lora / kv projection of an MLA-style attention layer: the q row
(1280 columns) is normed and quantized to MXFP8 for the `wq_b` GEMMs, the kv row (512 columns) is normed to bf16 for
the RoPE and cache insert. Inputs bf16 `[M, 1280]` and `[M, 512]` with unit column stride and any row stride (the two
halves of one `[M, 1792]` row), bf16 weights, fp32 eps (1e-20 in the served model), M >= 0 (decode rows 1-768, any
larger M works). Outputs: e4m3 codes `[M, 1280]` and E8M0 scale bytes `[M, 40]`, both row-major (the layout the vLLM
MXFP8 linear takes as a pre-quantized `QuantizedActivation`), and bf16 `[M, 512]`.

Contract (`reference.py`): each norm is `bf16(x * rsqrt(mean(x^2) + eps) * w)` in fp32; the quantizer then reads that
bf16 value (never the fp32 one) as vLLM's Triton `mxfp8_e4m3_quantize` does: per 32 columns
`clamp(ceil(log2(amax / 448)) + 127, 0, 254)` and e4m3 codes of `x * 2^(127 - byte)`. The kernel's square sum is the
served Triton `fused_q_kv_rmsnorm`'s reduction tree (8 warps of 4 columns, DPP wave fold, warp tree), so every output
is bit-identical to the served norm + quantizer (tests: 20 row counts to 2048, 3 seeds each); torch's sum differs in
the last fp32 bit, so against the torch reference about 3e-6 of the bf16 values differ by an ulp.

## Entry points

- `rmsnorm_dual_mxfp8`: the q codes + scales and the kv bf16 of a q / kv pair; outputs may be the caller's static
  buffers; graph-capture safe, no atomics, bitwise repeatable.
- `warmup_rmsnorm_dual_mxfp8`: compiles the one variant before graph capture (also the first call's warm-up).

The kernel has one variant at every M (`kernel.py`: seven waves per row, five for q and two for kv).

## Performance

One MI355X, graph replay of 100 distinct inputs, median of 5 x 5 replays, rounds A B B A, us per call. Baseline: vLLM's
Triton `fused_q_kv_rmsnorm` + `mxfp8_e4m3_quantize` of q (two launches); AITER's `fused_dual_rmsnorm_mxfp8_quant` is
one launch that quantizes the fp32 norm (not bit-compatible with the served path). This entry is bit-compatible and as
fast: both sit within 0.3 us of the launch floor up to 256 rows.

| rows | 16 | 32 | 64 | 96 | 128 | 192 | 256 | 384 | 512 | 768 |
|---|---|---|---|---|---|---|---|---|---|---|
| Triton norm + quant | 3.93 | 4.04 | 4.12 | 4.40 | 4.46 | 4.51 | 4.68 | 4.92 | 5.23 | 6.50 |
| AITER fused | 2.07 | 2.06 | 2.15 | 2.44 | 2.51 | 2.54 | 2.58 | 2.83 | 3.14 | 4.02 |
| this entry | 2.08 | 2.08 | 2.15 | 2.31 | 2.40 | 2.46 | 2.49 | 2.83 | 3.19 | 3.74 |

## Tests

```bash
python3 -m pytest tests/norm/rmsnorm_dual_mxfp8_q1280_kv512 -q
python3 benchmarks/norm/rmsnorm_dual_mxfp8_q1280_kv512/benchmark_norm.py
```

One GPU: every decode row count against the reference and the served Triton path, extreme rows (zero, overflow,
subnormal), caller buffers, graph replay with poisoned outputs, determinism, no compile after the warm-up.
