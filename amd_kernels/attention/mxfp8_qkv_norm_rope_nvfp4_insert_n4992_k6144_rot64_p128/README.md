# QKV MXFP8 GEMM + norm / RoPE + NVFP4 page inserts, N 4992, K 6144, rotary 64, pages 128 (TP2)

## What it does

The QKV chain of a MiniMax-M3.1-Flash TP2 sparse attention layer in one entry: the MXFP8 QKV GEMM (N 4992 = 32 q + 2
k + 2 v + 2 index-q + 1 replicated index-k heads of 128, K 6144), then per head the Gemma RMS norm and partial NeoX
RoPE (rotary 64), bf16 q `[m, 4096]`, e4m3 index-q `[m, 256]` (row stride divisible by 4), and the NVFP4 K / V /
index-K inserts into `[blocks, 4, 128, 72]` main and `[blocks, 128, 72]` index pages (bytes of skipped and unwritten
tokens stay). Input: the producer's e4m3 codes in the (16, 16) tile (`prequant_code_tile_rows(m, 4992, 6144)` rows,
1-768), shuffled E8M0 scales and the `preshuffle_mxfp8_weight` weight; heads 32 / 2 / 2, `eps >= 2^-126`. Bytes: bit
for bit the deferred chain at every row count (`gemm.mxfp8_common.mxfp8_gemm_shuffled(defer_reduce=True)` +
`splitk_reduce_norm_rope_nvfp4_insert`, or the bf16 GEMM + `norm_rope_nvfp4_insert`). One launch at 1-16 rows; at
17-768 the deferred chain's two launches: split-K partials at 17-48, the split-K decode GEMM (its K split reduced in
the kernel) at 49-256, the GEMM's bf16 stores non-temporal at 257-448 rows (the insert reads them at once). Contract:
`reference.py` (the GEMM reference, then the insert's).

## Entry points

- `mxfp8_qkv_norm_rope_nvfp4_insert`: the whole chain into the caller's q / index-q outputs and pages (slot -1
  skips); M 0 is a no-op, rows outside the shuffled-code tile's 1-768 raise `ValueError`. An `amd_kernels.warm_up`
  entry: the first call compiles rows 1-768 and must be eager (capture raises).

Usual caller: `mxfp8_gemm_qkv_norm_rope_nvfp4_insert` of
[`norm_rope_nvfp4_insert_d128_rot64_p128`](../norm_rope_nvfp4_insert_d128_rot64_p128/README.md), which routes the
TP2 heads here (`one_launch=False` keeps its two launches). Counters: one set per eager stream or per (capture ID,
stream), from a pool of 512 sets (2.4 MiB) per device. `host.py` picks the one launch, the chain and its store policy
by row count.

## Performance

Graph replay, us per call, layer 39 TP2 rank 0 weights in 14 cold copies, up to 100 inputs (decode rows: requests of
8 tokens), one MI355X. Deferred chain: the same GEMM row + the TP2 insert in two launches. Stock: vLLM's TP2 path
(Triton `tl.dot_scaled` + vLLM's FlyDSL insert). Roofline: 6.5 TB/s over weight, codes and insert bytes, or 5 PF/s.

| M | entry | deferred chain | stock | roofline |
|---|---|---|---|---|
| 1 / 8 / 16 (one launch) | 9.01 / 9.26 / 9.52 | 9.34 / 10.17 / 10.27 | 12.94 / 13.56 / 14.12 | 4.87 / 4.88 / 4.90 |
| 32 (chain) | 11.03 | 11.03 | 15.00 | 4.94 |
| 64 / 96 / 128 / 176 / 256 (chain, split-K decode GEMM) | 12.79 / 13.96 / 14.97 / 17.34 / 20.32 | 12.80 / 13.95 / 14.97 / 17.36 / 20.32 | 18.35 / 23.45 / 25.10 / 27.42 / 41.02 | 5.02 / 5.09 / 5.17 / 5.28 / 5.47 |
| 288 / 352 / 384 / 416 (chain, nt GEMM stores) | 24.01 / 25.02 / 25.43 / 28.01 | 24.34 / 25.61 / 26.01 / 28.16 | 37.89 / 39.01 / 40.92 / 63.59 | 5.54 / 5.69 / 5.77 / 5.84 |
| 512 / 768 (chain) | 30.46 / 36.18 | 30.42 / 36.19 | 70.95 / 72.91 | 6.28 / 9.42 |

## Tests

```bash
python3 -m pytest tests/attention/mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128 -q
python3 benchmarks/attention/mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128/benchmark_chain.py \
    --checkpoint <M3.1-Flash checkpoint> --out /tmp/qkv_fused_tp2.json
```
