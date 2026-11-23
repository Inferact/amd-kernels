# Kernels

One line per kernel package. Each package's README has what it does, its entry points, performance and tests.

## Op entries

Callers (vLLM) import ops from `amd_kernels.<category>` only, never a `*_common` module or a shape package by path;
each category's `__init__.py` re-exports its families' `__all__` and holds no logic. The convention
([`routes.py`](routes.py)):

- An op lives in a `<category>/<family>_common` package: `<op>(...)` reads its shape key from its tensors and calls
  the shape package its `Routes` table names (imported on first use); an uncovered key raises `ValueError`.
- `<op>_blocker(...)` takes the op's arguments (same names and order; arguments it does not read may default to None,
  facts the tensors do not show are extra keyword-only arguments) and reads host metadata only: None when a package
  serves the call and beats the served path at its row count (`Rows`), else the reason the caller logs before its own
  path. The op itself still runs outside those rows (tests, benchmarks). `<op>_shapes()`: the covered keys.
- An op that binds a weight or builds state at load time has `prepare_<op>(...)`: a `Prepared` handle (weights in the
  kernel's layout from the checkpoint's, workspaces, communicators; compiled) whose `blocker` / `try_call` take the
  call's arguments, or the reason. Load-time checks without an op are `<thing>_layer_blocker`, `_weight_blocker`,
  `_config_blocker`, ...
- Previous names (`*_unsupported_reason`, integer-argument blockers) answer as before for one tag and warn once
  (`renamed`, `previous_form`); they are not in `__all__`.

[`tests/test_entry_convention.py`](../tests/test_entry_convention.py) checks the convention for every category.

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
- [`gemm/bf16_skinny_common`](gemm/bf16_skinny_common/README.md): the dense bf16 linears of a 1-192 row decode step
  (attention in/out projections, shared experts, latent down, lm_head, drafter context projection) on a load-time
  packed weight, one tuned tile table.
- [`gemm/bf16_common`](gemm/bf16_common/README.md): bf16 GEMM entries (`bf16_latent_down_mxfp4_quant`,
  `bf16_linear_gather`): the shape -> package routing and the reason a call has no kernel.
- [`gemm/bf16_n20480_k7168`](gemm/bf16_n20480_k7168/README.md): the TP8 lm_head vocabulary shard, with a row map and a
  publish epilogue into the logits exchange.
- [`gemm/bf16_latent_down_mxfp4_quant_k7168_n3584`](gemm/bf16_latent_down_mxfp4_quant_k7168_n3584/README.md): the
  latent MoE down projection (7168 -> 3584) with the routed experts' MXFP4 activation quant fused into its epilogue.
- [`gemm/bf16_linear_gather_n896_k35840_ranks8`](gemm/bf16_linear_gather_n896_k35840_ranks8/README.md): the drafter's
  context projection as an eight-rank N-shard GEMM with the column gather and RMSNorm in one launch.
- [`gemm/bf16_gemm_common`](gemm/bf16_gemm_common/README.md): bf16 GEMM entries (`bf16_gemm` for lm_head / vocabulary
  shards, `bf16_gemm_f32_bf16`): the shape -> package routing, the lm_head row ranges and `prepare_bf16_gemm`.

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
- [`moe/latent_moe_common`](moe/latent_moe_common/README.md): latent MoE decode entries (`router_sigmoid_bias_topk`,
  `mxfp4_moe_experts`, `allreduce_latent_rmsnorm_up_proj`, `layer_tail_launch`, ...): the shape -> package routing and
  the reason a call has no kernel.
- [`moe/mxfp4_h3584_i3072_e896_top16`](moe/mxfp4_h3584_i3072_e896_top16/README.md): routed MXFP4 experts (A4W4, SiTUv2)
  of one TP8 rank: activation quant, GEMM1 + GEMM2 in one launch, deterministic top-16 sum.
- [`moe/router_sigmoid_bias_top16_e896_h7168`](moe/router_sigmoid_bias_top16_e896_h7168/README.md): the MoE router,
  bf16 gate GEMM + sigmoid + correction bias + top-16 + renormalisation, two launches.
- [`moe/allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8`](moe/allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8/README.md):
  the MoE tail on eight ranks: latent all-reduce + RMSNorm + the hidden-shard up projection of the shared output.
- [`moe/layer_tail_k2_h7168_ranks8`](moe/layer_tail_k2_h7168_ranks8/README.md): one decode layer's whole tail in one
  persistent kernel on eight ranks: o_proj, AttnRes, router, shared + routed MXFP4 experts, latent norm, up projection.
- [`moe/router_topk_common`](moe/router_topk_common/README.md): the sqrt-softplus top-k router entry (`router_topk`,
  `prepare_router_topk`).
- [`moe/mxfp4_experts_common`](moe/mxfp4_experts_common/README.md): MXFP8 x MXFP4 routed-expert entries
  (`mxfp4_experts`, `expert_parallel`, `prepare_expert_parallel`).
- [`moe/mxfp4_moe_common`](moe/mxfp4_moe_common/README.md): the MXFP4 MoE entry of the 6144-hidden TP shards
  (`mxfp4_moe`, `prepare_mxfp4_moe_weights`).

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
- [`attention/norm_rope_insert_common`](attention/norm_rope_insert_common/README.md): RMS norm + RoPE + KV insert
  entries (`qkv_norm_rope_insert`, `kv_norm_rope_insert_multilayer`): the shape -> package routing and the reason a
  call has no kernel.
- [`attention/norm_rope_insert_d64_rot64`](attention/norm_rope_insert_d64_rot64/README.md): the Kimi-K3.1 draft's
  RMS norm + NeoX RoPE + bf16 / fp8 KV insert of 64-dim heads: one layer's qkv rows, the context rows of all layers.
- [`attention/mla_decode_common`](attention/mla_decode_common/README.md): DSA / MLA decode entries
  (`qkv_a_norm_kv_insert`, `indexer_prep`, `q_bproj_wuk_gather`, `fp8_sparse_mla_decode`, `lse_combine_vup`,
  `fp8_dense_mla_decode`): the shape -> package routing and the reason a call has no kernel.
- [`attention/dsa_prologue_qkv_a_norm_kv_insert_h7168`](attention/dsa_prologue_qkv_a_norm_kv_insert_h7168/README.md):
  the DSA decode prologue around the qkv_a GEMM: q_a / kv_a norms + fp8_ds_mla KV insert, and the indexer's key
  LayerNorm + MXFP4 insert, query MXFP4 quant and head weights.
- [`attention/dcp8_q_bproj_wuk_gather_h96_kv512`](attention/dcp8_q_bproj_wuk_gather_h96_kv512/README.md): the DSA
  decode query path at DCP8: q_b projection, W_UK absorption and the push all-gather of the query over the eight ranks,
  as bf16 and/or the hi/lo fp8 record the sparse-MLA decode kernel takes.
- [`attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048`](attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048/README.md):
  the DSA layers' sparse MLA decode at DCP8: 96 latent heads over an fp8_ds_mla cache (bf16 or pre-quantized q), bf16
  output and LSE of the rank's key share.
- [`attention/dcp8_lse_combine_vup_mla_h12_kv512`](attention/dcp8_lse_combine_vup_mla_h12_kv512/README.md): the DCP8
  MLA decode combine: W_UV up-projection, owner-rank push and LSE-weighted combine in one launch.
- [`attention/fp8_dense_mla_decode_h96_kv512`](attention/fp8_dense_mla_decode_h96_kv512/README.md): the DSpark draft's
  dense MLA decode (96 latent heads, e4m3 q and latent KV, causal round-robin DCP shard).
- [`attention/kda_common`](attention/kda_common/README.md): KDA entries (`kda_decode_fb_gate_spec`, `kda_prefill`,
  `kda_mono_decode`): the shape -> package routing and the reason a call has no kernel.
- [`attention/kda_decode_fb_gate_spec_h12_d128`](attention/kda_decode_fb_gate_spec_h12_d128/README.md): the
  linear-attention (KDA) spec-verify decode: f_b gate GEMM, conv + SiLU, delta rule with per-token state snapshots and
  the gated RMSNorm in one launch.
- [`attention/kda_prefill_h12_d128`](attention/kda_prefill_h12_d128/README.md): the linear-attention (KDA) chunked
  prefill: conv + SiLU, gate and q / k norms, the chunk-group delta-rule scan with fp32 recurrent states, the gated
  RMSNorm (one launch for short batches, staged launches for long ones).
- [`attention/kda_mono_decode_k1_h12_d128`](attention/kda_mono_decode_k1_h12_d128/README.md): one persistent launch of
  a KDA layer's decode (AttnRes, in_proj, conv, delta-rule recurrence, gated norm) for a spec-verify step of up to 16
  rows.
- [`attention/sparse_mla_decode_common`](attention/sparse_mla_decode_common/README.md): the sparse MLA decode entry.
- [`attention/rope_fp8_kv_insert_common`](attention/rope_fp8_kv_insert_common/README.md): the RoPE + fp8 KV insert
  entry.
- [`attention/compressor_insert_common`](attention/compressor_insert_common/README.md): the compressor insert entry.
- [`attention/gemma_norm_rope_insert_common`](attention/gemma_norm_rope_insert_common/README.md): the Gemma-norm qkv
  norm + RoPE + fp8 insert entry.
- [`attention/norm_rope_nvfp4_insert_common`](attention/norm_rope_nvfp4_insert_common/README.md): the norm + RoPE +
  NVFP4 insert entries (with and without the MXFP8 QKV GEMM).
- [`attention/nvfp4_sparse_attention_common`](attention/nvfp4_sparse_attention_common/README.md): NVFP4 sparse
  attention entries (decode, index + decode, prefill) and the layer object (`prepare_nvfp4_sparse_layer`).
- [`attention/swa_decode_common`](attention/swa_decode_common/README.md): the sliding-window decode entry (o_proj
  input).

## Indexer

- [`indexer/indexer_nvfp4_d128_p128_top16`](indexer/indexer_nvfp4_d128_p128_top16/README.md): the sparse-attention
  indexer: block scores over NVFP4 index pages and the top-16 blocks, decode and prefill, one GPU at 1 / 2 index heads
  (TP4 / TP2), or two or four GPUs at decode.
- [`indexer/indexer_mxfp4_h32_d128_top512`](indexer/indexer_mxfp4_h32_d128_top512/README.md): MXFP4 indexer top-512
  over decode rows without writing the logits.
- [`indexer/q_rope_mxfp4_quant_h32_d128_rot64`](indexer/q_rope_mxfp4_quant_h32_d128_rot64/README.md): indexer query
  RoPE + MXFP4 quant, 32 heads of 128.
- [`indexer/mxfp4_indexer_topk_common`](indexer/mxfp4_indexer_topk_common/README.md): the MXFP4 top-512 indexer and
  query RoPE + MXFP4 entries (`prepare_mxfp4_indexer_topk`).
- [`indexer/nvfp4_indexer_common`](indexer/nvfp4_indexer_common/README.md): the NVFP4 prefill top-16 indexer entry.

## Engram

- [`engram/ngram_rows_mxfp8_d256_c24_ranks8`](engram/ngram_rows_mxfp8_d256_c24_ranks8/README.md): n-gram hash and
  MXFP8 row pull from the owner ranks' tables, eight ranks.
- [`engram/gated_post_bf16_h5120_hc4`](engram/gated_post_bf16_h5120_hc4/README.md): gated Engram update of the four
  residual streams, bf16.
- [`indexer/mxfp4_indexer_common`](indexer/mxfp4_indexer_common/README.md): MXFP4 DSA indexer entries
  (`mxfp4_paged_score_topk`, `mxfp4_prefill_score`, `mxfp4_prefill_topk`): the shape -> package routing and the reason a
  call has no kernel.
- [`indexer/mxfp4_paged_score_topk_d128_h32_top2048`](indexer/mxfp4_paged_score_topk_d128_h32_top2048/README.md):
  the DSA decode indexer: MXFP4 paged MQA scores of 32 heads and each row's top-2048 keys, the logits never stored.
- [`indexer/mxfp4_prefill_score_d128_h32`](indexer/mxfp4_prefill_score_d128_h32/README.md): the DSA prefill indexer's
  MXFP4 MQA logits of ragged rows (32 heads, head dim 128) over staged key pages, eight rows per workgroup sharing each
  key page.
- [`indexer/mxfp4_prefill_topk_d128_h32_top2048`](indexer/mxfp4_prefill_topk_d128_h32_top2048/README.md): the DSA
  prefill indexer's exact top-2048 select over each row's ragged key window of the FP4 scorer's logits.
- [`engram/engram_common`](engram/engram_common/README.md): the n-gram row pull and gated update entries.

## Embedding and activation

- [`embedding/vocab_gather_common`](embedding/vocab_gather_common/README.md): the vocabulary gather entry
  (`vocab_gather_comm`): (hidden, world size) -> package, two, four or eight shards.
- [`embedding/vocab_gather_h6144`](embedding/vocab_gather_h6144/README.md): vocabulary lookup over two or four shards
  without an all-reduce.
- [`embedding/vocab_gather_h7168_v163840_ranks8`](embedding/vocab_gather_h7168_v163840_ranks8/README.md): vocabulary
  lookup over eight shards without an all-reduce.
- [`embedding/vocab_gather_gemma_mxfp8_h6144`](embedding/vocab_gather_gemma_mxfp8_h6144/README.md): the same lookup +
  Gemma norm + MXFP8 quant in one launch.
- [`activation/swiglu_oai_mxfp8_i3072`](activation/swiglu_oai_mxfp8_i3072/README.md): SwiGLU + MXFP8 quant for the
  TP4 down GEMM (I 3072).
- [`activation/swiglu_oai_mxfp8_i6144`](activation/swiglu_oai_mxfp8_i6144/README.md): SwiGLU + MXFP8 quant for the
  TP2 down GEMM (I 6144).
- [`activation/clamp_silu_mul_mxfp8_i2304`](activation/clamp_silu_mul_mxfp8_i2304/README.md): clamped SiLU-mul + MXFP8
  quant for the shared expert's down GEMM.
- [`activation/clamp_silu_mul_mxfp8_common`](activation/clamp_silu_mul_mxfp8_common/README.md): the clamped SiLU-mul +
  MXFP8 entry.

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
- [`norm/attnres_rmsnorm_common`](norm/attnres_rmsnorm_common/README.md): the all-reduce + AttnRes + RMSNorm entry
  (`attnres_comm`): (hidden, world size) -> package and the reason a call has no kernel.
- [`norm/allreduce_attnres_rmsnorm_h7168_ranks8`](norm/allreduce_attnres_rmsnorm_h7168_ranks8/README.md): eight-rank
  all-reduce + AttnRes block mix + RMSNorm of a decode step (hidden 7168, bf16).
- [`norm/rmsnorm_mxfp8_common`](norm/rmsnorm_mxfp8_common/README.md): RMSNorm + MXFP8 and dual RMSNorm entries.
- [`norm/mhc_common`](norm/mhc_common/README.md): the mHC seam, finisher and tail entries (seam row ranges).
- [`norm/gemma_add_rmsnorm_common`](norm/gemma_add_rmsnorm_common/README.md): Gemma add-RMSNorm + MXFP8 and the
  all-reduce norm / dense MLP entries (`prepare_decoder_ar_norm`, `prepare_ar_norm_dense_mlp`).

## Collectives (eight ranks)

- [`collectives/handshake_ranks8`](collectives/handshake_ranks8/README.md): the eight-rank flag barrier and small
  payload exchanges (epoch-tagged words, flag after data, pull) on the shared helpers in `common/`.
- [`collectives/logits_exchange_common`](collectives/logits_exchange_common/README.md): the lm_head shard GEMM + logits
  exchange entry (`logits_exchange_comm`): (N shard, K, world size) -> the exchange and GEMM packages it composes and
  the reason a call has no kernel.
- [`collectives/all_to_all_bf16_n20480_ranks8`](collectives/all_to_all_bf16_n20480_ranks8/README.md): logits rows into
  the owner's `[rows, 163840]` layout (or an all-gather).
- [`collectives/topk_gather_merge_common`](collectives/topk_gather_merge_common/README.md): the DCP top-k gather + merge
  entry (`topk_gather_merge_comm`): (top-k, world size) -> package and the reason a call has no kernel.
- [`collectives/dcp8_topk_gather_merge_top2048`](collectives/dcp8_topk_gather_merge_top2048/README.md): the DSA score
  all-gather, global top-2048 merge and this rank's CSR KV slots in one launch.

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
- [`collectives/seq_parallel_handoffs_common`](collectives/seq_parallel_handoffs_common/README.md): the
  sequence-parallel handoff entry (`prepare_seq_parallel_handoffs`).

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
- [`sampling/markov_gumbel_common`](sampling/markov_gumbel_common/README.md): the fused Markov draft chain entry
  (`markov_gumbel_fused`): (vocabulary, rank) -> package and the reason a call has no kernel.
- [`sampling/markov_gumbel_fused_v163840_r512`](sampling/markov_gumbel_fused_v163840_r512/README.md): the DSpark
  draft's Markov chain in one persistent launch: bias GEMV, logits cache, Gumbel draw, argmax, k steps.
- [`sampling/top_p_mask_common`](sampling/top_p_mask_common/README.md): the top-p mask entry (`top_p_mask`):
  vocabulary (163840, 200064) -> package and the reason a call has no kernel.
- [`sampling/top_p_mask_f32_v200064`](sampling/top_p_mask_f32_v200064/README.md): vLLM's top-p without top-k on fp32
  logits rows, bit for bit: the split chain (<= 64 rows) in one launch, the one-program-per-row kernel above.
- [`sampling/markov_chain_gumbel_bf16_v129280_r256_s5`](sampling/markov_chain_gumbel_bf16_v129280_r256_s5/README.md):
  5-step Markov draft chain for every request of the batch in one launch.
- [`sampling/rejection_standard_v129280_s5`](sampling/rejection_standard_v129280_s5/README.md): standard rejection
  sampling of 5 draft tokens, two launches.
- [`sampling/top_p_mask_f32_v163840`](sampling/top_p_mask_f32_v163840/README.md): top-p mask of <= 64 fp32 logits rows,
  vocabulary 163840: vLLM's split top-p chain in one launch.
- [`sampling/markov_chain_common`](sampling/markov_chain_common/README.md): the one-launch Markov draft chain entry.
- [`sampling/rejection_sample_common`](sampling/rejection_sample_common/README.md): the rejection sampling entry.
- [`sampling/batch_sharded_sampling_common`](sampling/batch_sharded_sampling_common/README.md): the batch-sharded
  sampling entry (`prepare_batch_sharded_sampling`).

## Decoder blocks

- [`decoder/dense_mxfp8_h6144_i3072_ranks4`](decoder/dense_mxfp8_h6144_i3072_ranks4/README.md): the draft's dense
  decoder layers around their attention, four ranks.
- [`decoder/dense_mxfp8_h6144_i6144_ranks2`](decoder/dense_mxfp8_h6144_i6144_ranks2/README.md): the draft's dense
  decoder layers around their attention, two ranks (and the TP2 target's dense MLP).
- [`decoder/bf16_dense_mlp_common`](decoder/bf16_dense_mlp_common/README.md): the dense bf16 MLP shard entry
  (`bf16_dense_mlp`): (hidden, inter) -> package and the reason a call has no kernel.
- [`decoder/bf16_dense_mlp_h7168_i1792`](decoder/bf16_dense_mlp_h7168_i1792/README.md): the DSpark drafter's dense
  MLP shard (gate_up + SwiGLU, down) on packed bf16 operands.
- [`decoder/dense_mxfp8_decoder_common`](decoder/dense_mxfp8_decoder_common/README.md): the dense MXFP8 decoder
  entry (`prepare_dense_decoder`).

## Shared code

- [`warm_up.py`](warm_up.py): entry registration and the first-call warm-up.
- [`compile_watch.py`](compile_watch.py): reports a compile after the warm-ups.
- [`common/`](common/): buffer ops, launch helpers, the AITER GEMM's XCD swizzle, the HIP runtime binding, the
  eight-rank flag helpers (`rank8_signal.py`, `rank8_comm.py`) and the persistent-kernel building blocks (`mono/`).
