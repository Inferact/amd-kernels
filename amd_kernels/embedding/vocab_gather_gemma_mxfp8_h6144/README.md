# Peer vocabulary lookup + Gemma RMSNorm + MXFP8 quant on two or four ranks, hidden 6144

## What it does

One launch for the owner embedding lookup of `embedding/vocab_gather_h6144` followed by
`norm/gemma_rmsnorm_bf16_mxfp8_h6144`'s plain Gemma norm and MXFP8 quant: it returns the same codes and shuffled
scales, and the raw bf16 embedding rows as the residual of the first post-attention tail. It reads the peer weights
through a live `VocabGather` mapping (no second weight copy, no communication buffers, no handshake, no output
atomics). Shapes: two or four equal contiguous immutable bf16 vocabulary shards, hidden 6144; 0-2048 contiguous int32
/ int64 ids; norm weight bf16 `[6144]`, epsilon a runtime fp32. Outputs: e4m3 codes `[ceil(rows / tile) * tile, 6144]`
in the 16x16 tiled layout (tile 0: row-major `[rows, 6144]`), E8M0 scales `[ceil(padded_rows / 32) * 32, 192]` in
AITER's shuffled layout, residual `[rows, 6144]` bf16; padding unwritten. Caller: the MiniMax-M3.1 DSpark draft's
layer 0 (embedding + no-residual input norm + quant, the codes feed layer 0's QKV GEMM), through `embed_norm_quant`
of `decoder/dense_mxfp8_h6144_i3072_ranks4` (TP4) and `decoder/dense_mxfp8_h6144_i6144_ranks2` (TP2). Contract
(`reference.py`): the lookup's rule (invalid ids give zero rows; bf16 subnormals and signed zeros become +0), then the
plain norm with its 12 wave partials and FMA / reduction tree, rounded to bf16, then the per-32 E8M0 quantizer: codes
and scales equal lookup + plain norm bit for bit; norm accuracy follows the plain norm's fp64 contract.

## Entry points

- `vocab_gather_gemma_mxfp8`: codes, scales and the raw bf16 residual of a batch of ids, eager or in a graph, through
  a live `embedding.vocab_gather_h6144.VocabGather`; the code tile is a multiple of 16 (the consumer GEMM's
  `prequant_code_tile_rows`) or 0. Keep the mapping and its weight alive and unchanged through every graph replay,
  and close it collectively after all graphs finish.
- `warmup_vocab_gather_gemma_mxfp8`: compiles all 8 variants of the mapping's world size (2 wave forms x 2 id widths
  x 2 code layouts); call it before serving or graph capture.

Dispatch: `host.py` picks the 12-wave or four-wave kernel by row count, and the variant by the mapping's world size.

## Performance

MI355X, graph replay of 100 changing id batches (7 rows per request: six mask ids and one random anchor), warm
weights, max over ranks, median of alternated rounds, us per call. 2 ranks: 100032-row shards (one xGMI link), stock =
`VocabParallelEmbedding` (mask / gather / fill + AITER custom all-reduce) + `gemma_rmsnorm_mxfp8`
(`embedding/vocab_gather_h6144/benchmark_stock.py --norm`); 4 ranks: 51200-row shards, baseline = `VocabGather` +
`gemma_rmsnorm_mxfp8`.

| Requests (rows) | 1 (7) | 10 (70) | 16 (112) | 32 (224) | 64 (448) | 128 (896) | 256 (1792) |
|---|---|---|---|---|---|---|---|
| 2 ranks, stock + norm / quant | 27.56 | 41.57 | 51.54 | 79.74 | 133.30 | 240.20 | 442.99 |
| 2 ranks, fused | 4.48 | 5.88 | 6.65 | 9.36 | 16.38 | 30.91 | 54.57 |
| 4 ranks, lookup + norm / quant | 6.11 | 7.17 | 7.66 | 9.69 | 16.03 | 29.55 | 57.18 |
| 4 ranks, fused | 4.40 | 5.37 | 5.77 | 7.00 | 10.75 | 19.81 | 35.47 |

## Tests

```bash
python3 -m pytest tests/embedding/vocab_gather_gemma_mxfp8_h6144 -q  # 4 GPUs and 2 GPUs
python3 benchmarks/embedding/vocab_gather_gemma_mxfp8_h6144/benchmark_gather_norm.py --world 2  # or 4
```
