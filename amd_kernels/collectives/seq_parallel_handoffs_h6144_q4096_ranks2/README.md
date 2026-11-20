# Sequence-parallel layer hand-offs, hidden 6144, 4096 attention columns per rank, two ranks

## What it does

`SeqParallelHandoffs` runs the chains of a sequence-parallel (SP) decoder layer of MiniMax-M3.1-Flash TP2 prefill
between the head-sharded parts (qkv, attention, the FFN of every token) and the token-sharded residual stream, with the
four `pull_collectives_ranks2` communicators and the warm-up; the caller keeps the layer loop, attention, the first
and final norms and the DBO gating. o_proj_full MXFP8 `[6144, 8192]` (`gemm/mxfp8_common`), the MoE of
`moe/mxfp4_h6144_i1536_e128_top4`. Rank r holds `shard_rows(t)` tokens (a multiple of 32), 2048 <= t <= 32768.
Contract (`reference.py`): `pack_reference` bit for bit (the Triton `mxfp8_quant_pack` but for blocks with a maximum
below 2^-119); every phase output bit for bit its kernels' outputs on the rows the collectives deliver (gathers and
fp32 reduce-scatter sums in rank order); no atomics, no quantized reduction; the MoE tail's reduce-scatter crosses the
link in `pull_collectives_ranks2`'s exact 12-bit code (`coded_tail`). With the full shared expert copy each rank runs
it on its own rows (a side stream, joined by an event before the tail) and adds it to its own rows' slot sum; only
the routed payload is gathered (3296 B/token; a shard copy keeps the 7328 B compact one). Eager only (host-side
sequence state), on the current stream, the same calls on both ranks.

## Entry points

- `SeqParallelHandoffs`: a collective; builds the transport for a batch-token limit (clamped to 2048..32768) and
  compiles it; `supports` says whether a token count fits.
- `SeqParallelHandoffs.warm_up`: local, before serving: every compute variant at 2048-32768 tokens.
- `SeqParallelHandoffs.input_gather`: the first layer's normed bf16 shard -> MXFP8 codes + AITER-shuffled scales of
  every token.
- `SeqParallelHandoffs.attention_moe`: this rank's heads of every token -> the routed-only `MoeInput` of every token,
  the new residual shard and the `SharedPending` shared expert of the own rows; `attention_dense`: the same o_proj,
  then the dense MLP's codes + scales of every token.
- `SeqParallelHandoffs.ffn_norm_gather`: the FFN output of every token (MoE slots + `SharedPending` or a shared
  partial, or a dense partial) -> the next layer's codes, scales and residual shard.
- `SharedPending`: the shared expert's output and the event `wait` joins; `OWNER_SHARED`: the feature flag.
- `SeqParallelHandoffs.ffn_reduce_scatter`: after the last layer, this rank's summed rows; `gather`: every token's.
- `shard` / `shard_rows`: a rank's zero-padded rows of a full tensor / their count (a multiple of 32).
- `mxfp8_pack_q4096` (also `pack`): bf16 `[M, 4096]` -> uint8 `[M, 4224]` (4096 codes, 128 scale bytes per row).
- `SdmaSeqParallelHandoffs`: the same phases and bits on SDMA pushes (`sdma_push_ranks2`); each phase takes a hook
  that runs after a push and before its wait (DBO: the yield); one instance per microbatch; `drain` waits on the host
  for every push.

The phase order and kernels of each chain are in `handoffs.py` and `handoffs_sdma.py`.

## Performance

A MoE layer from the attention output to the next layer's QKV output (attention core not run), eager, rank max, median
of 7 rounds, us (`benchmark_sp_layer.py`); "shard copy": the shared expert's TP shard on every token (the compact
gather; under DBO on the caller's stream). TP: o_proj on every token + all-reduce (AITER custom up to 64 MiB, else
RCCL) + norm, MoE, all-reduce + norm/quant. DBO: two microbatches on the SDMA form, vLLM's turn-taking. Ideal: link
bytes (23456 B per shard row) at 61.2 GB/s plus (DBO: max with) the compute ideal. DBO and pull are even at 8192.

| T tokens | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|---:|
| SP pull (shard copy) | 1141 (1230) | 1750 (1942) | 3009 (3392) | 5510 (6335) | 10783 (12549) |
| SP SDMA DBO (shard copy) | - | 2178 (2355) | 2983 (3199) | 4778 (5185) | 8695 (9377) |
| TP2 (best all-reduce) | 1584 | 2814 | 5262 | 10048 | 19885 |
| best SP / ideal | 67% | 70% | 53% | 66% | 72% |

## Tests

```bash
python3 -m pytest tests/collectives/seq_parallel_handoffs_h6144_q4096_ranks2 -q  # 2 GPUs; test_warm_up.py alone
python3 benchmarks/collectives/seq_parallel_handoffs_h6144_q4096_ranks2/benchmark_sp_layer.py [out.json]
```
