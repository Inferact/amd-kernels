# Residual add + Gemma RMSNorm entries (one GPU and the TP norm tails), routed by shape

## What it does

Residual add + Gemma RMSNorm: on one GPU with the MXFP8 quant written in the code layout the next MXFP8 GEMM reads
(key: hidden from the norm weight; 6144 -> `gemma_add_rmsnorm_h6144`), and the TP communicators of a decoder layer's
two norm tails with the all-reduce in front and the FFN between them (`DecoderArNorm`; key (hidden, world) from the
norm weight and the CPU group: (6144, 4) -> `allreduce_add_rmsnorm`, (6144, 2) -> `allreduce_add_rmsnorm_ranks2`; a
dense layer's MLP by (hidden, I) of its down weight). The tails are the communicator's methods. Numerics are the
package's `reference.py`.

## Entry points

- `gemma_add_rmsnorm_common.available`: gfx950 with FlyDSL importable.
- `gemma_add_rmsnorm_common.gemma_add_rmsnorm_mxfp8_for`: the one-GPU op; `gemma_add_rmsnorm_mxfp8_for_blocker`,
  `gemma_add_rmsnorm_mxfp8_for_shapes`, `warm_up_gemma_add_rmsnorm_mxfp8_for`.
- `gemma_add_rmsnorm_common.decoder_ar_norm_world_blocker`: load-time check (world, hidden, IPC-shareable inputs).
- `gemma_add_rmsnorm_common.prepare_decoder_ar_norm`: the communicators or the reason (also `decoder_ar_norm`,
  `decoder_ar_norm_blocker`, `decoder_ar_norm_shapes`).
- `gemma_add_rmsnorm_common.prepare_ar_norm_dense_mlp`: a dense layer's MLP for `post_attention` (also
  `ar_norm_dense_mlp`, `ar_norm_dense_mlp_blocker`).

Dispatch: `_ROUTES` in `add_rmsnorm.py`, `_ROUTES` and `_MLP` in `allreduce.py`; a shape package is imported on
the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/norm/gemma_add_rmsnorm_common/test_routing.py -q   # host only
python3 -m pytest tests/norm/gemma_add_rmsnorm_common -q
```
