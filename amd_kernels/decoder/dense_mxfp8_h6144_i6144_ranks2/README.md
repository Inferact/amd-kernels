# Dense decoder layers around their attention on two TP ranks, MXFP8 linears, hidden 6144, I 6144 per rank

## What it does

A dense decoder layer stack without its attention, on two TP ranks: the input fc split by N (5 taps of 6144 ->
6144, 3072 columns per rank), the embedding owner read + Gemma norm + MXFP8 quant, the all-reduce + add + Gemma norm
(+ quant) tails, the MLP (gate_up [12288, 6144] -> SwiGLU-OAI -> down [6144, 6144], TP partial), the stack around the
caller's attention and the context K / V chain (Gemma norm + MXFP8 GEMM [2560, 6144] + multi-layer K norm, RoPE, fp8
insert into 2-KV-head caches). Callers: the MiniMax-M3.1 DSpark draft at TP2 and the target's dense layers 0-2
(`norm/gemma_add_rmsnorm_h6144/allreduce_add_rmsnorm_ranks2`, form `fp32`). Each call picks its consumer GEMM's code
layout (`gemm.mxfp8_common.gemm.prequant_code_tile_rows`) and row-count variant; weights are preshuffled MXFP8.
Contract (`reference.py`): compositions of the per-kernel references, cosine ~1; each call equals its per-kernel
chain bitwise.

## Entry points

- `DenseDecoderRanks2`: the layers' communicators, built collectively on both ranks after the weights load, before
  any capture; its `fc` / `fc_cat` (eager), `embed_norm_quant`, `ar_norm_quant`, `ar_norm`, `forward` (the stack
  around the caller's attention callback, 1-2048 ids), `store_context_kv`, `capture`, `close`.
- `stack_blocker`, `context_kv_blocker`: why a model's layer stack or context K / V chain does not fit (None: fits).
- `mxfp8_swiglu_oai_mlp`: the MLP's down output from the gate_up input codes, 1-2048 rows, `bf16_chain` or `fp32`;
  `mlp_supported`: whether the MLP weights fit.
- `gemma_rmsnorm_mxfp8_linear`: Gemma norm -> MXFP8 codes in the GEMM's layout -> the GEMM, any row count and N.

The MLP and the norm + GEMM are `amd_kernels.warm_up` entries; the row-count dispatch lives in `mlp.py`.

## Performance

The whole draft step at TP2 (fc + embedding + 5 layers with the draft attention q32 / kv2 + final norm), 7 rows per
request, MAX over 2 ranks, median us (`benchmark_draft_forward.py`): the block's fc eager + its stack in one graph vs
the TP2 stock path (replicated fc, `VocabParallelEmbedding`, Triton norms and `tl.dot_scaled` linears, AITER custom
all-reduce) in one graph. Roofline: serial sum of max(HBM 6.5 TB/s, MXFP8 5 PF / bf16 2.5 PF) per op + the fc
exchange and ten tails over the one xGMI link at 56 GB/s (its link part in brackets).

| requests (rows) | 1 (7) | 8 (56) | 16 (112) | 32 (224) | 40 (280) | 48 (336) | 52 (364) | 64 (448) | 128 (896) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock TP2 | 656 | 1014 | 1262 | 1956 | 2232 | 2466 | 2600 | 2965 | 5222 |
| this block | 370 | 574 | 801 | 1149 | 1329 | 1551 | 1624 | 1878 | 3341 |
| roofline (link) | 180 (16) | 282 (102) | 403 (204) | 644 (409) | 765 (511) | 886 (613) | 946 (665) | 1127 (818) | 2223 (1636) |

## Tests

```bash
python3 -m pytest tests/decoder/dense_mxfp8_h6144_i6144_ranks2 -q  # test_decoder_ranks2.py: 2 GPUs
python3 benchmarks/decoder/dense_mxfp8_h6144_i6144_ranks2/benchmark_draft_forward.py  # 2 GPUs
```
