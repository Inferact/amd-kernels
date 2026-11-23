# Compressor tail and indexer key store, latent 512, key 128, rotary 64

## What it does

For a compress-ratio 1 or 2 layer, one launch takes the compressor GEMM's fp32 rows (`[kv | score]` `[rows, 1024]` for
ratio 2, `kv` `[rows, 512]` for ratio 1) to the layer's two caches: the closing rows of a group (`(position + 1) %
ratio == 0`, ring slot >= 0) are pooled (ratio 2: softmax over the previous row and this one, the previous row of a
request's first token from the ring), RMS-normed to a bf16 latent, RoPE'd (dims 448..511, rotary position of the group's
first token) and inserted as e4m3 (`/ kv_scale`) into the main cache `[pages, block, 512]`; the latent times `wk [128,
512]` is rounded to bf16, RMS-normed, RoPE'd (dims 64..127, the indexer's table) and inserted as MXFP4 into the paged
indexer cache in AITER's MQA-logits layout (68 bytes a token: 64 values, 4 e8m0 scales). Ratio 2 also saves the raw rows
of each request's last ring-size tokens into the ring. Slots < 0 skip; the ring, main and indexer caches are updated in
place. It replaces five launches (save + pool + norm, rope + quant insert, the wk GEMM, the key norm / RoPE / MXFP4
store). Contract: `reference.py`, bit for bit except about 1 indexer byte in 2e4 where the fp32 MFMA sum of `wk` rounds
to bf16 differently from the contract's exact sum; the vLLM / AITER chain agrees with the kernel on every byte in the
benchmark inputs (the tests allow a few: that chain sums its squares in the Triton tree order).

## Entry points

- `compressor_insert`: pool, norm, insert the main and indexer rows, save the ring.
- `shuffle_wk`: the load-time layout of `wk` (call once, before graph capture).
- `supports_compressor_insert`: whether the inputs fit (gfx950, fp32 raw rows, e4m3 main pages, 68-byte indexer tokens
  in pages of whole 32-token tiles).

An entry (`amd_kernels.warm_up`): the first call per (ratio, ring rows, main rows, indexer rows, table dtype) compiles
its kernel. The row count is a runtime argument: 16 tokens a workgroup, no dispatch.

## Performance

One MI355X; one graph of up to 100 launches with their own rows, slots and positions, decode batches of 6-row requests
padded to the row count, A B B A, median of 5 x 10 replays, us per call. The baseline is the chain above, the wk GEMM
included, not the kv_score GEMM (see `gemm/bf16_k5120_n1056_f32_n32`). The kernel is latency bound, not byte bound
(two memory round trips, four workgroup barriers; 2 us is the launch floor).

| Rows | Ratio 2: kernel | Baseline | Ratio 1: kernel | Baseline |
|---|---|---|---|---|
| 16 | 9.55 | 11.33 | 7.68 | 9.11 |
| 48 | 10.63 | 14.32 | 8.45 | 9.51 |
| 96 | 10.82 | 14.75 | 8.51 | 10.02 |
| 192 | 11.02 | 14.91 | 8.65 | 10.62 |
| 384 | 11.24 | 15.25 | 8.73 | 10.85 |
| 768 | 13.28 | 18.08 | 9.51 | 11.87 |
| 848 | 13.60 | 18.35 | 9.77 | 12.17 |
| 1024 | 13.94 | 18.87 | 10.03 | 12.63 |

## Tests

```bash
python3 -m pytest tests/attention/compressor_insert_d512_idx128_rot64 -q
python3 benchmarks/attention/compressor_insert_d512_idx128_rot64/benchmark_compressor_insert.py
```
