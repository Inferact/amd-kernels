# Kernels

One line per kernel package. Each package's README has what it does, its entry points, performance and tests.

## Op entries

Callers use one shape-agnostic entry per op ([`routes.py`](routes.py), issue #11): it reads the shape from its
tensors and calls the shape package; `<op>_blocker(...)` returns None or the reason a call is not covered (the caller
falls back), `<op>_shapes()` the covered shape keys.

- `gemm.mxfp8_common.gemm`: `mxfp8_linear`, `mxfp8_gemm` (+ `mxfp8_linear_blocker`, `tuned_shapes`) over the
  `gemm/mxfp8_n<N>_k<K>` folders in `gemm._SHAPE_HOSTS`.
- `gemm`: `mxfp8_gate_up_swiglu_mxfp8[_prequant]` (gate_up + clamped SwiGLU + MXFP8).
- `gemm`: `mxfp8_postmix_gemm`, `mxfp8_postmix_gates_gemm` (MXFP8 GEMM + mHC post-mix epilogue).
- `gemm`: `mxfp8_grouped_gemm`, `mxfp8_grouped_gemm_mxfp8` (grouped MXFP8 GEMM, bf16 or MXFP8 out).
- `gemm`: `bf16_gemm` (lm_head), `bf16_gemm_f32_bf16` (fp32 + bf16 columns in one GEMM).
- `attention`: `sparse_mla_decode`, `rope_fp8_kv_insert`, `compressor_insert`.
- `indexer`: `mxfp4_indexer_topk`, `mxfp4_indexer_candidate_topk`, `q_rope_mxfp4_quant`.
- `moe`: `router_topk`, `mxfp4_experts` (+ `mxfp4_expert_weights`), `expert_parallel` (fused EP, a collective).
- `norm`: `rmsnorm_mxfp8`, `rmsnorm_dual_mxfp8`, `mhc_post_pre_rmsnorm`, `mhc_post_collapse_rmsnorm`, `mhc_gate_finish`.
- `activation`: `clamp_silu_mul_mxfp8`.
- `sampling`: `markov_chain`, `rejection_sample`.
- `engram`: `ngram_rows`, `gated_post`.

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
- [`gemm/mxfp8_n25600_k6144`](gemm/mxfp8_n25600_k6144/README.md): the rows of the 25600 x 6144 GEMM (Engram `wkv`).
- [`gemm/mxfp8_gateup_swiglu_n6144_k6144_i3072`](gemm/mxfp8_gateup_swiglu_n6144_k6144_i3072/README.md): dense MLP
  gate_up + SwiGLU at 1-8 rows (TP4).
- [`gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144`](gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144/README.md): dense MLP
  gate_up + SwiGLU at 1-16 rows (TP2).
- [`gemm/mxfp8_bmm_mxfp8_out_n1024_k4096`](gemm/mxfp8_bmm_mxfp8_out_n1024_k4096/README.md): DeepSeek-V4.1's grouped
  `wo_a` with its bf16 output quantized to MXFP8 for `wo_b`, in one launch.
- [`gemm/bf16_n50016_k6144`](gemm/bf16_n50016_k6144/README.md): the TP4 lm_head vocabulary shard, with an optional row
  map.
- [`gemm/bf16_n100032_k6144`](gemm/bf16_n100032_k6144/README.md): the TP2 lm_head vocabulary shard, with an optional
  row map, a strided output and the exchange publish.
- [`gemm/bf16_n200064_k256`](gemm/bf16_n200064_k256/README.md): the draft's Markov projection over 0-128 selected rows.
- [`gemm/bf16_concat_linear_n1536_k30720`](gemm/bf16_concat_linear_n1536_k30720/README.md): the draft's input fc over
  five taps at TP4, and the four-rank gather.
- [`gemm/bf16_concat_linear_n3072_k30720`](gemm/bf16_concat_linear_n3072_k30720/README.md): the draft's input fc over
  five taps at TP2, and the two-rank exchange.
- [`gemm/mxfp8_n1792_k5120`](gemm/mxfp8_n1792_k5120/README.md): fused wq_a + wkv projection, N 1792 x K 5120, decode
  rows, quantization in the GEMM.
- [`gemm/mxfp8_n32768_k1280`](gemm/mxfp8_n32768_k1280/README.md): the wq_b projection, N 32768 x K 1280, decode rows.
- [`gemm/mxfp8_n4096_k1280`](gemm/mxfp8_n4096_k1280/README.md): the indexer wq_b projection, N 4096 x K 1280, decode
  rows.
- [`gemm/mxfp8_n4608_k5120`](gemm/mxfp8_n4608_k5120/README.md): the shared-expert gate_up projection, N 4608 x K 5120,
  decode rows.
- [`gemm/mxfp8_n5120_k2304`](gemm/mxfp8_n5120_k2304/README.md): the shared-expert down projection, N 5120 x K 2304,
  decode rows.
- [`gemm/mxfp8_n5120_k8192`](gemm/mxfp8_n5120_k8192/README.md): the wo_b projection, N 5120 x K 8192, decode rows.
- [`gemm/mxfp8_n5120_k15360`](gemm/mxfp8_n5120_k15360/README.md): the draft's input projection over three tapped
  hidden states, N 5120 x K 15360.
- [`gemm/mxfp8_decode_prequant`](gemm/mxfp8_decode_prequant/README.md): decode-row MXFP8 GEMM on pre-quantized rows,
  with a clamped swiglu + MXFP8 epilogue.
- [`gemm/mxfp8_grouped_g8_n1024_k4096`](gemm/mxfp8_grouped_g8_n1024_k4096/README.md): grouped MXFP8 GEMM, 8 groups x N
  1024 x K 4096 (the low-rank output projection's first GEMM), bf16 or MXFP8 out.
- [`gemm/mxfp8_postmix_hc4_n5120_k8192`](gemm/mxfp8_postmix_hc4_n5120_k8192/README.md): MXFP8 GEMM N 5120 x K 8192
  with the four-stream mHC post-mix epilogue.
- [`gemm/mxfp8_postmix_add_hc4_n5120_k2304`](gemm/mxfp8_postmix_add_hc4_n5120_k2304/README.md): MXFP8 GEMM N 5120 x K
  2304 with the routed add and the mHC post-mix epilogue.
- [`gemm/bf16_k5120_n1056_f32_n32`](gemm/bf16_k5120_n1056_f32_n32/README.md): bf16 GEMM, K 5120: fp32 compressor
  columns plus 32 bf16 weight columns, one launch.
- [`gemm/bf16_n129280_k5120`](gemm/bf16_n129280_k5120/README.md): the full-vocabulary lm_head, N 129280 x K 5120,
  bf16.

## MoE

- [`moe/mxfp4_h6144_i768_e128_top4`](moe/mxfp4_h6144_i768_e128_top4/README.md): the TP4 MoE block (I 768 per rank):
  routed MXFP4 experts (A4W4) with the MXFP8 shared expert.
- [`moe/mxfp4_h6144_i1536_e128_top4`](moe/mxfp4_h6144_i1536_e128_top4/README.md): the TP2 MoE block (I 1536 per rank):
  routed MXFP4 experts (A4W4) with the MXFP8 shared expert (a TP shard, or the full copy: its shard read in place,
  the whole expert on a sequence-parallel rank's own rows).
- [`moe/mxfp4_h5120_i2304_e48_top6`](moe/mxfp4_h5120_i2304_e48_top6/README.md): routed MXFP8 x MXFP4 experts, 48 local
  experts, top 6, H 5120 I 2304.
- [`moe/mxfp4_h5120_i2304_e16_top3`](moe/mxfp4_h5120_i2304_e16_top3/README.md): routed MXFP8 x MXFP4 experts, 16 local
  experts, top 3, H 5120 I 2304.
- [`moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8`](moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8/README.md):
  one-launch expert-parallel MoE on eight ranks: dispatch, experts, combine.
- [`moe/router_sqrtsoftplus_top6_h5120_e384`](moe/router_sqrtsoftplus_top6_h5120_e384/README.md): router: gate GEMM
  and biased sqrt-softplus top-6 of 384 experts.
- [`moe/router_sqrtsoftplus_top3_h5120_e128`](moe/router_sqrtsoftplus_top3_h5120_e128/README.md): router: gate GEMM
  and biased sqrt-softplus top-3 of 128 experts.

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
- [`attention/rope_fp8_kv_insert_h64_d512_rot64`](attention/rope_fp8_kv_insert_h64_d512_rot64/README.md): query RoPE +
  e4m3 quant and KV RoPE + e4m3 cache insert, 64 heads of 512.
- [`attention/compressor_insert_d512_idx128_rot64`](attention/compressor_insert_d512_idx128_rot64/README.md):
  compressor pool + RMSNorm + RoPE and the latent / indexer key cache inserts.
- [`attention/sparse_mla_decode_fp8kv_h64_d512_rot64`](attention/sparse_mla_decode_fp8kv_h64_d512_rot64/README.md):
  sparse decode attention, 64 heads x d512, e4m3 KV, inverse-RoPE MXFP8 output.

## Indexer

- [`indexer/indexer_nvfp4_d128_p128_top16`](indexer/indexer_nvfp4_d128_p128_top16/README.md): the sparse-attention
  indexer: block scores over NVFP4 index pages and the top-16 blocks, decode and prefill, one GPU at 1 / 2 index heads
  (TP4 / TP2), or two or four GPUs at decode.
- [`indexer/indexer_mxfp4_h32_d128_top512`](indexer/indexer_mxfp4_h32_d128_top512/README.md): MXFP4 indexer top-512
  over decode rows without writing the logits.
- [`indexer/q_rope_mxfp4_quant_h32_d128_rot64`](indexer/q_rope_mxfp4_quant_h32_d128_rot64/README.md): indexer query
  RoPE + MXFP4 quant, 32 heads of 128.

## Engram

- [`engram/ngram_rows_mxfp8_d256_c24_ranks8`](engram/ngram_rows_mxfp8_d256_c24_ranks8/README.md): n-gram hash and
  MXFP8 row pull from the owner ranks' tables, eight ranks.
- [`engram/gated_post_bf16_h5120_hc4`](engram/gated_post_bf16_h5120_hc4/README.md): gated Engram update of the four
  residual streams, bf16.

## Embedding and activation

- [`embedding/vocab_gather_h6144`](embedding/vocab_gather_h6144/README.md): vocabulary lookup over two or four shards
  without an all-reduce.
- [`embedding/vocab_gather_gemma_mxfp8_h6144`](embedding/vocab_gather_gemma_mxfp8_h6144/README.md): the same lookup +
  Gemma norm + MXFP8 quant in one launch.
- [`activation/swiglu_oai_mxfp8_i3072`](activation/swiglu_oai_mxfp8_i3072/README.md): SwiGLU + MXFP8 quant for the
  TP4 down GEMM (I 3072).
- [`activation/swiglu_oai_mxfp8_i6144`](activation/swiglu_oai_mxfp8_i6144/README.md): SwiGLU + MXFP8 quant for the
  TP2 down GEMM (I 6144).
- [`activation/clamp_silu_mul_mxfp8_i2304`](activation/clamp_silu_mul_mxfp8_i2304/README.md): clamped SiLU-mul + MXFP8
  quant for the shared expert's down GEMM.

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
- [`norm/rmsnorm_mxfp8_h5120`](norm/rmsnorm_mxfp8_h5120/README.md): RMSNorm + MXFP8 quant, H 5120.
- [`norm/rmsnorm_dual_mxfp8_q1280_kv512`](norm/rmsnorm_dual_mxfp8_q1280_kv512/README.md): dual RMSNorm: q row to
  MXFP8, kv row to bf16.
- [`norm/mhc_post_pre_rmsnorm_h5120`](norm/mhc_post_pre_rmsnorm_h5120/README.md): mHC seam in one launch: post-mix,
  gates, collapse, RMSNorm (+ MXFP8).
- [`norm/mhc_gate_finish_rmsnorm_h5120`](norm/mhc_gate_finish_rmsnorm_h5120/README.md): mHC seam finisher: GEMM tile
  records to gates, collapse and RMSNorm (+ MXFP8).
- [`norm/mhc_post_collapse_rmsnorm_h5120`](norm/mhc_post_collapse_rmsnorm_h5120/README.md): mHC step-end tail:
  post-mix, collapse and final RMSNorm.

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
- [`sampling/markov_chain_gumbel_bf16_v129280_r256_s5`](sampling/markov_chain_gumbel_bf16_v129280_r256_s5/README.md):
  5-step Markov draft chain for every request of the batch in one launch.
- [`sampling/rejection_standard_v129280_s5`](sampling/rejection_standard_v129280_s5/README.md): standard rejection
  sampling of 5 draft tokens, two launches.

## Decoder blocks

- [`decoder/dense_mxfp8_h6144_i3072_ranks4`](decoder/dense_mxfp8_h6144_i3072_ranks4/README.md): the draft's dense
  decoder layers around their attention, four ranks.
- [`decoder/dense_mxfp8_h6144_i6144_ranks2`](decoder/dense_mxfp8_h6144_i6144_ranks2/README.md): the draft's dense
  decoder layers around their attention, two ranks (and the TP2 target's dense MLP).

## Shared code

- [`warm_up.py`](warm_up.py): entry registration and the first-call warm-up.
- [`compile_watch.py`](compile_watch.py): reports a compile after the warm-ups.
- [`common/`](common/): buffer ops, launch helpers, the AITER GEMM's XCD swizzle, the HIP runtime binding.
