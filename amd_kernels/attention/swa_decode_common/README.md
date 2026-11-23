# Sliding-window decode attention entry (e4m3 K / V pages), routed by shape

## What it does

Decode attention of up to a few query rows per request over a sliding window of an e4m3 paged cache, written as the
o_proj's input: the MXFP8 epilogue (codes, scales, tile) where the package writes it for `oproj_n`, else bf16 shaped
like q. The key (head_dim, query heads per KV head) is read from the cache [pages, KV heads, page, 2 * head_dim] and q
([rows, H, d] or [rows, H * d]); (128, 16) -> `swa_decode_fp8kv_gqa16_d128`. Request count, window, causality and page
size are runtime or compile modes. Numerics are the package's `reference.py`; the entry returns the direct call's
bits. A call past the kernel's case (query rows, window, layout) gets a reason and the caller runs its own attention.

## Entry points

- `swa_decode_common.available`: gfx950 with FlyDSL importable.
- `swa_decode_common.swa_decode_layer_blocker`: load-time check of a layer (head dim, heads, window).
- `swa_decode_common.swa_decode_max_query_len`: the most query rows per request a layer's kernel takes.
- `swa_decode_common.swa_decode_oproj_input`: the attention as o_proj input; `swa_decode_oproj_input_blocker`,
  `swa_decode_oproj_input_shapes`.

Dispatch: `_ROUTES` in `decode.py`; a shape package is imported on the first call of its key.

## Performance

The entry adds only the dispatch (a table lookup and the host checks of `*_blocker`) in front of the routed package's
call; no new measurement. The package's README holds its kernel times.

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/attention/swa_decode_common/test_routing.py -q   # host only
python3 -m pytest tests/attention/swa_decode_common -q
```
