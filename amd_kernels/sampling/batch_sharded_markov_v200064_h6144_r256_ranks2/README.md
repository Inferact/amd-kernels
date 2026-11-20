# Batch-sharded sampling on two ranks: owned-row logits, output gather, Gumbel Markov draft

## What it does

Batch-sharded sampling on two TP ranks: each rank verifies, and drafts for, the requests it owns (batch position i on
rank i % 2, local row i // 2). Vocabulary 200064 (lm_head shards of 100032 in `shuffle_weight`'s layout), hidden 6144,
Markov rank 256 (`markov_w1` / `markov_w2` bf16 [200064, 256]); target 1-2048 logits rows, 2-256 requests, 1-8 logits
rows per request; the Markov draft 1-7 steps, up to 128 requests per rank. Caller: the MiniMax-M3.1 TP2 sampler and
DSpark Markov draft. Contract (`reference.py`): the per-kernel entries' references; the block equals the per-kernel
chain bit for bit (draft ids, every draft-cache byte, the logits). The owned rows' logits come from the vocabulary
shard exchange up to 96 rows per call and above from two local GEMMs (this rank's shard and a copy of the peer's)
with the same bits: one xGMI link carries M x 100 KB per direction, more than a second 1.23 GB weight read costs from
about 100 rows.

## Entry points

- `BatchShardedSampling`: the per-process object (communicators, owner tables, scratch), built collectively before
  any CUDA graph capture; with `lm_weight` (the shuffled shard every call passes) it copies the peer's shard over IPC
  (+1.23 GB per GPU); without it every call takes the exchange.
- `BatchShardedSampling.target_logits`: the `[owned rows, 200064]` bf16 logits of this rank's requests.
- `BatchShardedSampling.gather_sampler_output`: tokens and sampled / rejected / NaN counts of every request in batch
  order.
- `BatchShardedSampling.setup_markov` (collective, before capture) and `markov_draft`: the draft ids of every request
  in batch order and this rank's draft-cache rows; graph safe.
- `sampling_blocker`, `markov_blocker`: why the block or the draft cannot serve a shape (None: it can).
- `owner_tables`, `OwnerTables`: the index tables of one request count; `close`: collective teardown.

Every rank makes every call with the same counts; `target_logits` and `markov_draft` are `amd_kernels.warm_up` entries.

## Performance

Two GPUs (one xGMI link) of one MI355X node. `benchmark_block.py`: graph replay of 12 distinct inputs, five alternated
rounds, MAX over ranks, median µs, dense owners. Target: the verify logits of R requests x 8 rows; vLLM = vLLM's
TP2 path (its lm_head GEMM over all rows + RCCL all-gather into [M, 200064] on both ranks), vLLM a2a = upstream
batch-sharded (the same GEMM + RCCL all-to-all). Draft: `markov_draft` (draft lm_head of 7 R rows, seven Markov steps,
id gather) against only the draft logits of vLLM's replicated draft (its GEMM + all-gather, before its Markov loop).

| R | target M | vLLM | vLLM a2a | exchange | local | draft rows | vLLM logits | exchange | local |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 253.1 | 238.7 | **213.4** | 356.0 | 7 | 244.1 | **389.8** | 538.1 |
| 4 | 32 | 370.5 | 309.3 | **247.0** | 363.8 | 28 | 354.4 | **426.5** | 548.3 |
| 8 | 64 | 548.2 | 416.1 | **308.8** | 367.8 | 56 | 510.3 | **506.9** | 561.3 |
| 12 | 96 | 673.4 | 488.7 | **372.7** | 373.8 | 84 | 635.0 | **549.7** | 574.5 |
| 16 | 128 | 816.7 | 557.7 | 431.2 | **380.4** | 112 | 743.0 | 604.7 | **580.5** |
| 24 | 192 | 1088.2 | 712.4 | 596.1 | **401.7** | 168 | 1001.3 | 780.8 | **605.1** |
| 32 | 256 | 1369.2 | 861.2 | 756.0 | **423.1** | 224 | 1232.2 | 1060.0 | **638.8** |
| 40 | 320 | 1626.1 | 971.7 | 1132.1 | **540.0** | 280 | 1447.7 | 1305.6 | **730.0** |
| 48 | 384 | 1914.0 | 1149.9 | 1325.7 | **595.3** | 336 | 1736.5 | 1347.7 | **830.6** |
| 52 | 416 | 2140.8 | 1290.1 | 1450.0 | **634.1** | 364 | 1823.5 | 1410.8 | **841.0** |

Bold: the path `host.py` takes (exchange up to 96 rows, local above).

## Tests

```bash
python3 -m pytest tests/sampling/batch_sharded_markov_v200064_h6144_r256_ranks2 -q                        # 2 GPUs
python3 benchmarks/sampling/batch_sharded_markov_v200064_h6144_r256_ranks2/benchmark_block.py [R ...]    # 2 GPUs
```
