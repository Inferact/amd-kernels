# NVFP4 sparse-layer attention input entries (norms + RoPE + NVFP4 inserts), routed by shape

## What it does

The attention input of an NVFP4 sparse attention layer in one call: q / k / index-q / index-k RMS norms, NeoX RoPE,
q (bf16) and index-q (e4m3) out, K / V and index-K into their NVFP4 pages; from the bf16 QKV rows or from the MXFP8
codes of the QKV GEMM's input (`gemm.mxfp8_common` layouts; one launch where the package has it). The key (head_dim,
rotary_dim, page) is read from `q_norm_weight`, `cos_sin_cache` and the KV cache [pages, 2 * KV heads, page, 72];
(128, 64, 128) -> `norm_rope_nvfp4_insert_d128_rot64_p128` (16 / 1 / 1 or 32 / 2 / 2 heads per rank). Numerics are
the package's `reference.py`; the entry writes the direct call's bits. A request's tokens take consecutive slots.

## Entry points

- `norm_rope_nvfp4_insert_common.available`: gfx950 with FlyDSL importable.
- `norm_rope_nvfp4_insert_common.norm_rope_nvfp4_insert_layer_blocker`: load-time check of a layer (heads, RoPE,
  index-q dtype, cos / sin cache).
- `norm_rope_nvfp4_insert_common.qkv_norm_rope_nvfp4_insert`: from bf16 QKV rows; `qkv_norm_rope_nvfp4_insert_blocker`,
  `qkv_norm_rope_nvfp4_insert_shapes`.
- `norm_rope_nvfp4_insert_common.mxfp8_gemm_qkv_norm_rope_nvfp4_insert`: the QKV GEMM from its MXFP8 input + the insert;
  `mxfp8_gemm_qkv_norm_rope_nvfp4_insert_blocker`.

Dispatch: `_ROUTES`, `_GEMM`, `_LAYER` in `qkv.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/attention/norm_rope_nvfp4_insert_common/test_routing.py -q   # host only
python3 -m pytest tests/attention/norm_rope_nvfp4_insert_common -q
```
