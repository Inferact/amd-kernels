# MXFP8 GEMM, N 5120 x K 15360 (the draft's input projection)

## What it does

`out [M, 5120] = x [M, 15360] @ W^T` for the DSpark draft's `main_proj`: x is the three tapped target hidden states
(3 x 5120) side by side, bf16, quantized per 1 x 32 block; W is the checkpoint's E4M3 codes with 32 x 32 E8M0 block
scales `[160, 480]` (the loader expands them to one scale row per output row and preshuffles the weight); out is bf16.
M = the draft's context rows (about 4-5 per request) or a prefill chunk, 1-32768. Contract (`reference.py`): both
operands dequantized exactly, fp32 accumulation, one bf16 rounding; cosine >= 0.99999 and relative L2 <= 1e-3, split-K
partials summed in a fixed order, so two runs match bit for bit.

Producer and consumer: x is the `[T, 15360]` concatenation of the three tapped target hidden states as the model runner
passes them (bf16; the GEMM's own quantizer reads it, no other conversion). The bf16 output goes unchanged into
[`norm/rmsnorm_mxfp8_h5120`](../../norm/rmsnorm_mxfp8_h5120/README.md), which applies `main_norm` and writes the MXFP8
codes and row-major scales that the draft layers' `wq_a | wkv` GEMMs take as a pre-quantized activation (tested:
`test_output_feeds_the_norm_and_quant`), so no standalone quantizer launch follows the projection.

Kernels: the LDS-staged split-K GEMM and the persistent 4-wave 256 x 256 GEMM of
[`gemm/mxfp8_common`](../mxfp8_common/README.md), with the row tiers of `host.DISPATCH` measured on this shape (sweeps
of its staged, AITER, decode and 4-wave rows at 32-8192 rows).

## Entry points

- Callers use the common entries: `gemm.mxfp8_common.gemm.mxfp8_linear` / `mxfp8_gemm` with `mxfp8_linear_blocker`;
  this folder is registered in `gemm._SHAPE_HOSTS` (`ROWS` = `DISPATCH`, `covers`).
- `MainProjWeights.from_block_scales`: the checkpoint's weight and block scales in the GEMM's layout; once, before graph
  capture.
- `main_proj`: the projection of a bf16 activation; graph-capture safe. An `amd_kernels.warm_up` entry: the first call
  compiles every row tier.
- `warmup_main_proj`: compiles every tier before graph capture. `main_proj_reference`: the torch contract.
  `config_for`: the row tier's kernel row. `available`: gfx950 with FlyDSL importable.

The dispatch is `host.DISPATCH` (row upper bound -> kernel row), read through `config_for`.

## Performance

One MI355X, graph replay, the weight in 5 pointer-distinct copies (> 450 MB), distinct activations per slot, rounds
A B B A, median us per call. Baseline: vLLM's `rocm_mxfp8_block32_gemm_quant` (AITER's preshuffled FlyDSL GEMM with its
32 x 32 block-scale kernel; at 32 rows the Triton packed GEMM with the quantizer fused), the served path.

| M | 32 | 64 | 128 | 160 | 192 | 256 | 320 | 384 | 512 | 640 | 768 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 18.9 | 24.0 | 29.5 | 34.8 | 37.0 | 40.8 | 46.9 | 49.3 | 60.5 | 82.3 | 79.6 | 108.5 | 167.0 | 346.7 | 551.7 |
| this entry | 22.7 | 24.6 | 30.2 | 34.8 | 38.9 | 40.4 | 46.7 | 49.2 | 57.7 | 72.0 | 76.7 | 92.1 | 159.1 | 274.4 | 493.2 |
| change | +20% | +3% | +3% | 0% | +5% | -1% | 0% | 0% | -5% | -13% | -4% | -15% | -5% | -21% | -11% |

Below 640 rows no row of these kernels beats the baseline by more than the run-to-run spread, and at 32 rows the
baseline is 20% faster: the entry is correct there, not faster.

## Tests

```bash
python3 -m pytest tests/gemm/mxfp8_n5120_k15360 -q
python3 benchmarks/gemm/mxfp8_n5120_k15360/benchmark_main_proj.py [--rows 160 640]
```

All row-tier edges and the served counts (160 / 192 / 320 / 384 / 640 / 768, 4097, 8192) against the reference with
random block-scale exponents, repeat runs, graph replay with new inputs, no compile after the warm-up.
