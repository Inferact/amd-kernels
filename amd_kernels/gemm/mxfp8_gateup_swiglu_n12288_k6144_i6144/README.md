# MXFP8 gate_up + SwiGLU-OAI, N 12288 x K 6144, I 6144 (dense MLP, TP2, 1-16 rows)

## What it does
`act [m, 6144] = SwiGLU-OAI(gate, up)` with `gate | up = x @ W^T` in one launch, for 1-16 rows (one 16-row code block,
the same cost at every row count) of the MiniMax-M3.1-Flash dense MLP at TP2. W is the per-rank dense gate_up weight
as the common dispatcher holds it: MXFP8 [12288, 6144], gate rows 0-6143 and up rows 6144-12287, (16, 16) codes and
AITER-layout scales (`preshuffle_mxfp8_weight`), no copy. x: E4M3 codes in the (16, 16) layout padded to 16 rows ([16,
6144]), E8M0 scales in AITER's layout ([>= 32, 192]). Each dot product rounds to bf16 before the activation; the bf16
output feeds the down linear (N 6144, K 6144) and its own quantizer. Callers: the TP2 dense decoder block (the
target's dense layers with form `fp32`, the DSpark draft's MLP with `bf16_chain`); above the fused row bound they run
gate_up `mxfp8_linear` + `activation/swiglu_oai_mxfp8_i6144` + down. Contract: `mxfp8_common/reference.py`
`mxfp8_gateup_swiglu_bf16_reference` (fp64 dots rounded to bf16, the caller's form): per-row cosine >= 0.9999, and the
bf16 bits of the decode GEMM's activation with the same K-wave count (`host.CONFIG`: 1 for `fp32`, 3 for
`bf16_chain`).

## Entry points
- `host.mxfp8_gateup_swiglu_bf16`: the bf16 [m, 6144] activation of 1-16 rows in one launch; the caller picks the
  numerics form (`fp32` or `bf16_chain`), never M.
- `host.warmup_mxfp8_gateup_swiglu_bf16`: compiles the forms; call it before graph capture.

`host.CONFIG` picks the kernel (K-waves, lookahead) by form.

## Performance
One MI355X. Graph replay, 6 rotated weight copies (> 900 MB), 100 inputs, both sides alternated, median us. Baseline:
what vLLM runs at TP2 without it, Triton `tl.dot_scaled` gate_up + `swiglu_oai_split` (fp32) or the aten chain
(bf16_chain) + the same down linear. Stream roofline (6.5 TB/s): fused kernel 78.0 MB = 12.0 us, FFN 116.9 MB = 18.0 us.

| M | fp32 FFN | baseline | bf16_chain FFN | baseline | fused kernel fp32 / bf16_chain |
|---|---|---|---|---|---|
| 1 | 22.49 | 29.39 | 22.91 | 41.23 | 14.82 / 15.29 |
| 4 | 22.80 | 29.67 | 23.22 | 49.35 | 14.84 / 15.30 |
| 7 | 23.36 | 30.14 | 23.78 | 49.76 | 14.86 / 15.32 |
| 8 | 23.55 | 30.44 | 23.98 | 50.58 | 14.86 / 15.31 |
| 14 | 25.30 | 33.28 | 25.78 | 53.44 | 14.88 / 15.34 |
| 16 | 25.52 | 33.71 | 26.00 | 53.26 | 14.88 / 15.36 |

## Tests
- `python3 -m pytest tests/gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144 -q`
- `python3 benchmarks/gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144/benchmark_gateup_swiglu_bf16.py [--sweep]`
