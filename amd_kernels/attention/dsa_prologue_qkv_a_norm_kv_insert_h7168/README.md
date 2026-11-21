# DSA decode prologue around the qkv_a GEMM: norms + fp8_ds_mla insert, indexer prep

## What it does

Two launches replace the ~9 small ops that run between the dense bf16 GEMMs of one DSA layer in decode (the qkv_a
GEMM, `q_b`, `wq_index` stay separate). Kimi K3.1 DSA is NoPE: k_pe is stored as is, there is no RoPE. Slot mappings are
already DCP-localized (a token another rank owns has slot -1): a slot < 0 writes no cache byte.

- `qkv_a_norm_kv_insert`: the qkv_a row `[q 1536 | kv 512 | k_pe 64]` bf16 -> `q_a` / `kv_a` RMSNorm outputs (bf16,
  fp32 accumulation, one rounding) and the fp8_ds_mla record of each owned token at `cache[slot // 64, slot % 64]`
  (512 e4m3 codes of the normed kv, 4 fp32 power-of-two tile scales, 64 bf16 k_pe; 656 bytes). With `index`, the row
  also carries the indexer's key and head weights (`wk_index` / `wv_index` weights concatenated onto qkv_a's: columns
  2112..2271) and the same launch does the key LayerNorm + MXFP4 insert and the head weights.
- `indexer_prep`: the `wq_index` output `[n, 32, 128]` -> MXFP4 payload `[n, 32, 64]` + AITER row-group scales
  `[n, 1, 4, 16, 4]`; the top-k buffer rows set to -1; given the `wk_index` / `wv_index` outputs (no `index` side in
  qkv_a): the key row -> LayerNorm (bf16 weight / bias) -> MXFP4 -> the indexer page slot (4096 payload + 256 e8m0
  scale bytes per 64-slot page, physical page stride >= 4352) and `weights = wv_index out .float() * weight_scale`.

Numerics contract (`reference.py`): the query payload / scales / weights and the top-k reset are bit-equal; norm outputs
are bit-equal except one-ulp moves from the summation order (>= 99.9% equal); fp8 codes within one step where the tile
scale matches; key payload follows the LayerNorm rounding. Equal to the stock AITER / Triton / C++ ops on random data.
Slots are int64, rows of `qkv_a` need a row stride divisible by 8 elements.

## Entry points

- `qkv_a_norm_kv_insert.qkv_a_norm_kv_insert`: norms + KV insert of one layer (without a cache it norms only); `index=`
  an `IndexSide` adds the indexer's key and weights.
- `qkv_a_norm_kv_insert.supports_qkv_a_norm_kv_insert`: whether the inputs fit (gfx950, bf16, 656-byte rows).
- `indexer_prep.indexer_prep`: the indexer's query quant, top-k reset and (key given) key store and weights of one
  layer, in the form `_fp4_indexer` takes as `q_payload`, `q_scales`, `q_weights` (starts: the decode `query_start_loc`).
- `indexer_prep.supports_indexer_prep`: whether the inputs fit (32 heads x 128, `[pages, 64, 68]` cache, top-k % 1024).

Both are entries (`amd_kernels.warm_up`): one variant per mode (insert / index side / key; top-k width), the row count
is a runtime argument.

## Performance

One MI355X; one graph of 100 launches with their own inputs (together larger than the MALL), min of 3 medians, us per
call. Ceiling = the kernel's skeleton (`benchmarks/.../skeleton.py`: same loads, stores, cache hints and grid, no
arithmetic); a bare graph launch is 1.57 us, a dependent HBM load 245 ns. Stock = AITER `rms_norm` x2 +
`concat_and_cache_mla` (3 launches); `layer_norm`, `store_keys`, `fill_`, weights product, starts clamp,
`quantize_queries` (6 launches). % = skeleton / kernel (100 = at the skeleton).

| M | qkv_a: us, % / stock | qkv_a + index side: us, % | indexer_prep: us, % / stock | queries only: us, % |
|---|---|---|---|---|
| 6 | 1.90, 93 / 5.59 | 1.92, 92 | 1.93, 90 / 12.74 | 1.92, 91 |
| 12 | 1.91, 93 / 5.83 | 1.98, 91 | 1.95, 90 / 13.03 | 1.94, 90 |
| 24 | 1.94, 92 / 5.90 | 1.99, 91 | 1.95, 90 / 13.40 | 1.94, 91 |
| 40 | 2.02, 92 / 6.07 | 2.07, 91 | 2.11, 100 / 15.39 | 2.00, 100 |
| 65 | 2.07, 92 / 6.33 | 2.19, 90 | 2.14, 100 / 15.67 | 2.06, 100 |
| 80 | 2.22, 92 / 6.50 | 2.31, 91 | 2.18, 100 / 16.46 | 2.08, 100 |
| 128 | 2.32, 94 / 6.74 | 2.36, 93 | 2.23, 100 / 18.16 | 2.13, 100 |
| 192 | 2.37, 95 / 6.96 | 2.42, 93 | 2.29, 100 / 21.11 | 2.28, 100 |

## Tests

```bash
python3 -m pytest tests/attention/dsa_prologue_qkv_a_norm_kv_insert_h7168 -q
python3 benchmarks/attention/dsa_prologue_qkv_a_norm_kv_insert_h7168/benchmark_prologue.py
```
