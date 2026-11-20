# Lightning indexer over NVFP4 index pages, head dim 128, 128-token pages, top-16 (gfx950)

## What it does
The MiniMax-M3.1 sparse-attention indexer: per (index head, query row, 128-token block) the max of `q . k` over the
row's visible tokens, then each row's top-16 blocks (score desc, ties to the larger id, -1 padded) as ids or with the
attend's page-16 table and context bound. Query e4m3 `[rows, heads, 128]`; NVFP4 index pages `[blocks,128,72]` uint8
(per token 64 B e2m1 codes, 8 B e4m3 scales) scored as `e4m3_rn_satfinite(e2m1 x scale)` on fp8 MFMA; sentinels 1e30
(init) / 1e29 (local blocks); <= 8192 blocks (1M tokens) per row. One GPU at 1 / 2 / 4 index-q heads (TP4 1, TP2 2),
or at decode four TP4 ranks (a head each; replicated or rank-owned cache: rank r blocks b % 4 == r, slot p // 4) or two
TP2 ranks (all four heads over blocks b % 2 == r, partial top-16 exchanged and merged; replicated cache).
Callers: the model's sparse attention layers, `attention/nvfp4_sparse_decode_gqa16_d128_p128_top16`. Contract
(`reference.py`): scores within 2e-4 of `max(|ref|, 1)` of fp64; sentinels, ids, tables, bounds exact (AITER's
`pa_sparse_block_topk`); routes bitwise equal.

## Entry points
- `decode_top16.IndexDecodeTop16`: a step's decode top-16 ids `[heads, rows, 16]` int32; built with the four (TP4) or
  two (TP2) ranks' gloo group a collective that takes the multi-GPU route by request count (one GPU per call: DBO).
- `prefill_top16.index_prefill_top16`: the top-16 ids of every ragged prefill row in one call (one-GPU scorer, then the
  ids-only top-k); `index_prefill_top16_blocker`: why a model's indexer does not fit; `warm_up`: compiles it all.
- `score_decode.nvfp4_index_score_top16`: one-GPU decode scores + top-16, two launches; `nvfp4_index_score_rows`: the
  scores alone, for a consumer that selects in its prologue (`topk.top16_indices_p128_plain_wave4`);
  `nvfp4_index_top16_from_scores`: the selection alone; `nvfp4_index_score_decode`: any heads x rows <= 32.
- `score_prefill.nvfp4_index_score_prefill`: fp32 scores of ragged prefill rows (`host.nvfp4_e4m3_lut`: its key
  table); `supports_nvfp4_index_score_prefill`: its shape gate.
- `topk.index_topk_decode_indices`, `topk.index_topk_prefill_indices`: top-16 ids of score rows; `index_topk_prefill`:
  ids + page table + context bounds (AITER's `pa_sparse_block_topk` outputs); `supports_index_topk`: the shape gate.
- `split4.host.init_rank_owned` / `get_rank_owned`: the four-GPU buffers per micro-batch (`score_topk`, `score_push`:
  the scores alone in rank-owned order); `split2.host.IndexSplit2`: the two-GPU buffers and `score_topk`.

Decode calls read lengths on the device (graph-safe at max_model_len). `nvfp4_index_score_top16`, `_rows`, both
`score_topk` and `index_prefill_top16` are `amd_kernels.warm_up` entries. Dispatch: `decode_top16.py` the route,
`score_decode/host.py` the scorer and selector by heads and requests, `split4/host.py` / `split2/host.py` the multi-GPU
kernels, `score_prefill/host.py` the tile by heads and work, `topk/host.py` the window tile by `max_seq_len`.

## Performance
Graph replay, host 1M, cold pages, us per call, max over ranks; AgentX = request i of N at the (i + 0.5) / N quantile
of AgentX c16 + c32 inputs; AgentX decode = the decode-time contexts (Tests) of c64 runs at 4 rows, c16 + c32 at 8.

| Route | Rows per request x requests | Context | us | Baseline (us) |
|---|---|---|---|---|
| `nvfp4_index_score_top16` | 8 x 1 / 8 x 4 / 4 x 1 | 32K, 192K | 5.83 / 7.43 / 5.72, 7.87 / 14.75 / 7.78 | `nvfp4_index_score_decode` + `index_topk_decode_indices` 8.23 / 9.38 / 8.06, 10.49 / 20.16 / 10.25 |
| `index_prefill_top16`, 1 head, four GPUs at once | M 4096 / 8192 / 16384 x 1 | prefix 0 / 8K / 32K / 128K / 256K | 11.2 / 17.6 / 35.5 / 74.6 / 126.7, 22.0 / 29.3 / 50.2 / 123.2 / 226.5, 44.7 / 57.7 / 88.8 / 241.3 / 435.3 | vLLM's in-tree scorer + top-k: 11.1 / 17.7 / 42.9 / 91.3 / 146.2, 21.8 / 29.4 / 58.9 / 140.6 / 244.4, 44.7 / 61.3 / 98.2 / 261.1 / 453.8 |
| `IndexDecodeTop16` four GPUs | 4 x 6 / 12 / 24 / 32, 8 x 5 / 12 / 16 | AgentX decode | 13.57 / 20.67 / 26.70 / 28.50, 16.20 / 23.06 / 23.60 | a four-rank flag wait, then the selector: 14.99 / 21.54 / 27.59 / 28.92, 17.53 / 23.79 / 23.99 |
| `nvfp4_index_score_top16`, 2 heads (TP2) | 8 x 1 / 4 / 16 / 32 / 48 / 64, 4 x 32 / 64 / 128 | 32K-192K log-uniform | 6.75 / 10.52 / 23.25 / 40.98 / 58.58 / 78.65, 39.97 / 74.46 / 141.49 | `nvfp4_index_score_decode` + `index_topk_decode_indices`: 8.44 / 14.38 / 32.23 / 61.50 / 95.21 / 120.05, 49.96 / 104.43 / 213.64 |
| `IndexDecodeTop16` two GPUs (TP2) + the decode attention (the decode block) | 8 x 8 / 12 / 16 / 24 / 32 / 48 | 128K-224K | 34.54 / 42.18 / 48.00 / 66.23 / 80.65 / 111.40 | one GPU (scorer + the attention's top-16): 36.34 / 47.99 / 57.52 / 82.16 / 105.16 / 147.74 |
| `index_prefill_top16`, 2 heads (TP2), scorer + top-k | M 4096 / 8192 / 16384 x 1 | prefix 0 / 56K / 112K | 14.3 / 110.6 / 386.0 | the scorer plan at the one-head thresholds: 15.7 / 109.7 / 386.4 |

## Tests
A decode-time contexts file (`--lens-json`): per request count N, the (i + 0.5) / N quantiles of decoding contexts.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 python3 -m pytest tests/indexer/indexer_nvfp4_d128_p128_top16 -q  # *_ranks4 / *_ranks2 skip on fewer cards
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_indexer.py [--prefill] [--heads 2]
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_index_decode_h2.py --ref base=<frozen copy> --ql 8  # TP2
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_rank_owned.py --ql 4 --decode '' --lens-json <your lengths.json> --reqs 8,32,64  # 4 GPUs
HIP_VISIBLE_DEVICES=0,1,2,3 python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode_ranks4.py --ql 4 --lens-json <your decode-time contexts.json> --reqs 4,6,8,12  # the decode block on each route (SPLIT_MIN_REQS)
HIP_VISIBLE_DEVICES=0,1 python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode_ranks2.py --ql 8 --lens-json <contexts.json> --reqs 8,10,12,16  # TP2 (SPLIT2_MIN_REQS)
python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_decode_topk.py --selector rank_owned --ref today=<frozen copy> --lens <your lengths.json> --reqs 32,48 --ql 4
```
