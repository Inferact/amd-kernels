# Kernels

One line per kernel package. Each package's README has what it does, its entry points, performance and tests.

## GEMM

- [`gemm/mxfp8_common`](gemm/mxfp8_common/README.md): MXFP8 dense GEMM entries (`mxfp8_linear`, `mxfp8_gemm`, ...),
  the shape -> kernel selection (TP4 and TP2 shards) and the shared kernels.
- [`gemm/mxfp8_n1280_k6144`](gemm/mxfp8_n1280_k6144/README.md): the rows of the 1280 x 6144 GEMM.
- [`gemm/mxfp8_n1536_k6144`](gemm/mxfp8_n1536_k6144/README.md): the rows of the 1536 x 6144 GEMM.
- [`gemm/mxfp8_n2304_k6144`](gemm/mxfp8_n2304_k6144/README.md): the rows of the 2304 x 6144 GEMM.
- [`gemm/mxfp8_n2560_k6144`](gemm/mxfp8_n2560_k6144/README.md): the QKV projection at TP4.
- [`gemm/mxfp8_n3072_k6144`](gemm/mxfp8_n3072_k6144/README.md): the rows of the 3072 x 6144 GEMM.
- [`gemm/mxfp8_n4608_k6144`](gemm/mxfp8_n4608_k6144/README.md): the rows of the 4608 x 6144 GEMM.
- [`gemm/mxfp8_n4992_k6144`](gemm/mxfp8_n4992_k6144/README.md): the QKV projection at TP2.
- [`gemm/mxfp8_n6144_k768`](gemm/mxfp8_n6144_k768/README.md): the rows of the 6144 x 768 GEMM.
- [`gemm/mxfp8_n6144_k1536`](gemm/mxfp8_n6144_k1536/README.md): the rows of the 6144 x 1536 GEMM.
- [`gemm/mxfp8_n6144_k2048`](gemm/mxfp8_n6144_k2048/README.md): the attention output projection at TP4.
- [`gemm/mxfp8_n6144_k3072`](gemm/mxfp8_n6144_k3072/README.md): the rows of the 6144 x 3072 GEMM.
- [`gemm/mxfp8_n6144_k4096`](gemm/mxfp8_n6144_k4096/README.md): the attention output projection at TP2 (decode and
  every prefill chunk).
- [`gemm/mxfp8_n6144_k6144`](gemm/mxfp8_n6144_k6144/README.md): the rows of the 6144 x 6144 GEMM (dense gate_up at
  I 3072, dense down projection at I 6144).
- [`gemm/mxfp8_n6144_k8192`](gemm/mxfp8_n6144_k8192/README.md): the full output projection of the sequence-parallel
  prefill.
- [`gemm/mxfp8_n12288_k6144`](gemm/mxfp8_n12288_k6144/README.md): the rows of the 12288 x 6144 GEMM.
- [`gemm/mxfp8_gateup_swiglu_n6144_k6144_i3072`](gemm/mxfp8_gateup_swiglu_n6144_k6144_i3072/README.md): dense MLP
  gate_up + SwiGLU at 1-8 rows (TP4).
- [`gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144`](gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144/README.md): dense MLP
  gate_up + SwiGLU at 1-16 rows (TP2).
- [`gemm/bf16_n50016_k6144`](gemm/bf16_n50016_k6144/README.md): the TP4 lm_head vocabulary shard, with an optional row
  map.
- [`gemm/bf16_n100032_k6144`](gemm/bf16_n100032_k6144/README.md): the TP2 lm_head vocabulary shard, with an optional
  row map, a strided output and the exchange publish.
- [`gemm/bf16_n200064_k256`](gemm/bf16_n200064_k256/README.md): the draft's Markov projection over 0-128 selected rows.
- [`gemm/bf16_concat_linear_n1536_k30720`](gemm/bf16_concat_linear_n1536_k30720/README.md): the draft's input fc over
  five taps at TP4, and the four-rank gather.
- [`gemm/bf16_concat_linear_n3072_k30720`](gemm/bf16_concat_linear_n3072_k30720/README.md): the draft's input fc over
  five taps at TP2, and the two-rank exchange.

## MoE

- [`moe/mxfp4_h6144_i768_e128_top4`](moe/mxfp4_h6144_i768_e128_top4/README.md): the TP4 MoE block (I 768 per rank):
  routed MXFP4 experts (A4W4) with the MXFP8 shared expert.
- [`moe/mxfp4_h6144_i1536_e128_top4`](moe/mxfp4_h6144_i1536_e128_top4/README.md): the TP2 MoE block (I 1536 per rank):
  routed MXFP4 experts (A4W4) with the MXFP8 shared expert (a TP shard, or the full copy: its shard read in place,
  the whole expert on a sequence-parallel rank's own rows).

## Attention

- [`attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16`](attention/nvfp4_sparse_prefill_gqa16_d128_p128_top16/README.md):
  sparse prefill attention over the NVFP4 KV cache, 1 or 2 KV heads per GPU (TP4 / TP2; bf16, fp16 or fp8 compute).
- [`attention/nvfp4_sparse_decode_gqa16_d128_p128_top16`](attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/README.md):
  sparse decode attention over the NVFP4 KV cache, 1 or 2 KV heads per GPU (TP4 / TP2; bf16, fp16 or fp8 compute).
- [`attention/norm_rope_nvfp4_insert_d128_rot64_p128`](attention/norm_rope_nvfp4_insert_d128_rot64_p128/README.md):
  q / k norm + partial RoPE + NVFP4 K / V / index-K inserts in one launch (q16 kv1 iq1 at TP4, q32 kv2 iq2 at TP2).
- [`attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128`](attention/mxfp8_qkv_norm_rope_nvfp4_insert_n2560_k6144_rot64_p128/README.md):
  the TP4 QKV GEMM, its reduce and the norm / RoPE / inserts, fused for small batches.
- [`attention/mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128`](attention/mxfp8_qkv_norm_rope_nvfp4_insert_n4992_k6144_rot64_p128/README.md):
  the TP2 QKV GEMM, its reduce and the norm / RoPE / inserts: one launch at 1-16 rows, the deferred chain to 768.
- [`attention/nvfp4_sparse_layer_gqa16_d128_p128_top16`](attention/nvfp4_sparse_layer_gqa16_d128_p128_top16/README.md):
  one NVFP4 sparse attention layer at its served dtypes, TP4 and TP2: the indexer, the warm-up, the decode / prefill
  calls.
- [`attention/swa_decode_fp8kv_gqa16_d128`](attention/swa_decode_fp8kv_gqa16_d128/README.md): the draft's
  sliding-window decode attention over an fp8 KV cache, 1 or 2 KV heads per GPU (TP4 / TP2).
- [`attention/norm_rope_fp8_insert_d128_rot64`](attention/norm_rope_fp8_insert_d128_rot64/README.md): the draft's
  norm + RoPE + fp8 KV insert, 1 or 2 KV heads per GPU (TP4 / TP2).

## Indexer

- [`indexer/indexer_nvfp4_d128_p128_top16`](indexer/indexer_nvfp4_d128_p128_top16/README.md): the sparse-attention
  indexer: block scores over NVFP4 index pages and the top-16 blocks, decode and prefill, one GPU at 1 / 2 index heads
  (TP4 / TP2), or two or four GPUs at decode.

## Embedding and activation

- [`embedding/vocab_gather_h6144`](embedding/vocab_gather_h6144/README.md): vocabulary lookup over two or four shards
  without an all-reduce.
- [`embedding/vocab_gather_gemma_mxfp8_h6144`](embedding/vocab_gather_gemma_mxfp8_h6144/README.md): the same lookup +
  Gemma norm + MXFP8 quant in one launch.
- [`activation/swiglu_oai_mxfp8_i3072`](activation/swiglu_oai_mxfp8_i3072/README.md): SwiGLU + MXFP8 quant for the
  TP4 down GEMM (I 3072).
- [`activation/swiglu_oai_mxfp8_i6144`](activation/swiglu_oai_mxfp8_i6144/README.md): SwiGLU + MXFP8 quant for the
  TP2 down GEMM (I 6144).

## Norms

- [`norm/gemma_add_rmsnorm_h6144`](norm/gemma_add_rmsnorm_h6144/README.md): add + Gemma RMSNorm to bf16 or MXFP8, with
  the four-rank (`allreduce_add_rmsnorm`) or two-rank (`allreduce_add_rmsnorm_ranks2`) all-reduce fused in front; a
  decoder layer's tail (norm, FFN, the next norm).
- [`norm/allreduce_gemma_bf16_mxfp8_h6144`](norm/allreduce_gemma_bf16_mxfp8_h6144/README.md): four-rank all-reduce +
  Gemma norm with bf16-rounded MXFP8.
- [`norm/allreduce_gemma_bf16_mxfp8_h6144_ranks2`](norm/allreduce_gemma_bf16_mxfp8_h6144_ranks2/README.md): two-rank
  all-reduce + Gemma norm with bf16-rounded MXFP8.
- [`norm/gemma_rmsnorm_bf16_mxfp8_h6144`](norm/gemma_rmsnorm_bf16_mxfp8_h6144/README.md): Gemma RMSNorm to
  bf16-rounded MXFP8.

## Collectives (four ranks)

- [`collectives/pull_collectives_ranks4`](collectives/pull_collectives_ranks4/README.md): the unpacking all-to-all,
  all-gather and fixed-order reduce-scatter over IPC-mapped pools.
- [`collectives/seq_parallel_handoffs_h6144_q2048_ranks4`](collectives/seq_parallel_handoffs_h6144_q2048_ranks4/README.md):
  the sequence-parallel prefill's layer hand-offs.
- [`collectives/sdma_push_ranks4`](collectives/sdma_push_ranks4/README.md): pushes on the SDMA engines, GPU-ordered,
  without CUs.
- [`collectives/all_to_all_bf16_n50016_ranks4`](collectives/all_to_all_bf16_n50016_ranks4/README.md): logits rows into
  the final vocabulary layout.
- [`collectives/all_gather_sampler_i64_c8_ranks4`](collectives/all_gather_sampler_i64_c8_ranks4/README.md): the
  sampler output gather.
- [`collectives/all_gather_i64_c7_ranks4`](collectives/all_gather_i64_c7_ranks4/README.md): draft token ids into batch
  order.

## Collectives (two ranks)

- [`collectives/pull_collectives_ranks2`](collectives/pull_collectives_ranks2/README.md): the unpacking all-to-all,
  all-gather, fixed-order reduce-scatter and SP norm tails over the one xGMI link of a GPU pair.
- [`collectives/seq_parallel_handoffs_h6144_q4096_ranks2`](collectives/seq_parallel_handoffs_h6144_q4096_ranks2/README.md):
  the sequence-parallel prefill's layer hand-offs (pull collectives, an SDMA form for DBO; the shared expert local to
  its rows' owner on a side stream).
- [`collectives/sdma_push_ranks2`](collectives/sdma_push_ranks2/README.md): pushes on one SDMA engine, GPU-ordered,
  without CUs.
- [`collectives/all_to_all_bf16_n100032_ranks2`](collectives/all_to_all_bf16_n100032_ranks2/README.md): logits rows
  into the final vocabulary layout.
- [`collectives/all_gather_sampler_i64_c8_ranks2`](collectives/all_gather_sampler_i64_c8_ranks2/README.md): the
  sampler output gather.
- [`collectives/all_gather_i64_c7_ranks2`](collectives/all_gather_i64_c7_ranks2/README.md): draft token ids into batch
  order.

## Sampling

- [`sampling/batch_sharded_markov_v200064_h6144_r256_ranks4`](sampling/batch_sharded_markov_v200064_h6144_r256_ranks4/README.md):
  batch-sharded sampling on four ranks: logits exchange, output gather and the Markov draft chain.
- [`sampling/batch_sharded_markov_v200064_h6144_r256_ranks2`](sampling/batch_sharded_markov_v200064_h6144_r256_ranks2/README.md):
  batch-sharded sampling on two ranks: owned-row logits (exchange, or locally on a copy of the peer shard), output
  gather and the Markov draft chain.
- [`sampling/gumbel_partials_bias_bf16_v200064`](sampling/gumbel_partials_bias_bf16_v200064/README.md): Gumbel noise
  and block partials of one Markov draft step.
- [`sampling/gumbel_combine_markov_embed_v200064_r256`](sampling/gumbel_combine_markov_embed_v200064_r256/README.md):
  the end of one Markov draft step: combine, draft id, next row.
- [`sampling/top_p_mask_f32_v200064`](sampling/top_p_mask_f32_v200064/README.md): vLLM's top-p without top-k on fp32
  logits rows, bit for bit: the split chain (<= 64 rows) in one launch, the one-program-per-row kernel above.

## Decoder blocks

- [`decoder/dense_mxfp8_h6144_i3072_ranks4`](decoder/dense_mxfp8_h6144_i3072_ranks4/README.md): the draft's dense
  decoder layers around their attention, four ranks.
- [`decoder/dense_mxfp8_h6144_i6144_ranks2`](decoder/dense_mxfp8_h6144_i6144_ranks2/README.md): the draft's dense
  decoder layers around their attention, two ranks (and the TP2 target's dense MLP).

## Shared code

- [`warm_up.py`](warm_up.py): entry registration and the first-call warm-up.
- [`compile_watch.py`](compile_watch.py): reports a compile after the warm-ups.
- [`common/`](common/): buffer ops, launch helpers, the AITER GEMM's XCD swizzle, the HIP runtime binding.
