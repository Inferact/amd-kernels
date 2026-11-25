# RMS norm + RoPE + KV insert entries, routed by shape

## What it does

One entry per op of the attention input of a GQA layer with plain RMS-normed q / k heads: per-head RMS norm, NeoX
RoPE and the K / V cache insert. A caller passes tensors; the entry reads the shape from them and calls the shape
package that serves it, so the caller imports no shape package:

| op | key, read from | shape packages |
|---|---|---|
| `qkv_norm_rope_insert` | (head_dim, rotary_dim) of `q_norm_weight` [head_dim] and `cos_sin_cache` [rows, rotary_dim] | `norm_rope_insert_d64_rot64` |
| `kv_norm_rope_insert_multilayer` | (head_dim, rotary_dim) of `k_norm_weight` [layers, head_dim] and `cos_sin_cache` | `norm_rope_insert_d64_rot64` |

Head counts, layers, page size and cache dtype are the package's compile modes, not keys. The Gemma-norm packages
(`norm_rope_fp8_insert_d128_rot64`) have another numerics contract and are not routed here. Arguments, dtypes and the
numerics contract are the routed package's (its `reference.py`); the entry's q and caches are bit for bit those of
the direct package call. For a call no package serves, `*_blocker` (the op's own arguments) returns a
one-line reason and the caller falls back; it reads only shapes, dtypes, strides and host values (no device read,
safe inside CUDA graph capture). The op itself only looks its key up and raises `ValueError` with the same "no kernel
for" text off the table.

## Entry points

- `norm_rope_insert_common.available`: gfx950 with FlyDSL importable (load-time check).
- `norm_rope_insert_common.norm_rope_insert_rope_blocker`: whether a layer's RoPE (head size, rotary dims, NeoX, cos / sin
  cache) fits a kernel of this family (the ops cannot see the RoPE style).
- `norm_rope_insert_common.qkv_norm_rope_insert`: q / k norm + RoPE of one layer's fused qkv rows into the caller's
  q, and the K / V insert of the tokens with slots; `qkv_norm_rope_insert_blocker`,
  `qkv_norm_rope_insert_shapes` (covered keys), `qkv_norm_rope_insert_kernel` (the routed warm-up entry).
- `norm_rope_insert_common.kv_norm_rope_insert_multilayer`: K norm + RoPE and the K / V inserts of the context rows
  into every layer's cache; `kv_norm_rope_insert_multilayer_blocker`, `kv_norm_rope_insert_multilayer_shapes`,
  `kv_norm_rope_insert_multilayer_kernel`.

Dispatch: `_ROUTES` (`amd_kernels.routes.Routes`) in `qkv.py`, `multilayer.py`; a shape package is imported on the first
call of its key.

## Performance

The entry adds only the dispatch: a CUDA graph of entry calls holds the same kernel as one of direct calls. MI355X,
graph of 10 calls on the same (warm) inputs, min of 15 replays, us per call; host us per eager call with the routed
function replaced by a no-op (entry minus direct), and of the reason the caller runs.

| call | direct graph us | entry graph us | dispatch host us | reason host us |
|---|---|---|---|---|
| qkv 32 rows, e4m3 | 2.96 | 2.94 | 0.25 | 3.12 |
| qkv 192 rows, e4m3 | 3.03 | 3.02 | 0.25 | 3.15 |
| multi-layer 40 rows, e4m3 | 2.94 | 2.94 | 0.26 | 8.35 |
| multi-layer 240 rows, e4m3 | 3.06 | 3.06 | 0.27 | 8.35 |

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/attention/norm_rope_insert_common/test_routing.py -q   # host only
python3 -m pytest tests/attention/norm_rope_insert_common -q
python3 benchmarks/attention/norm_rope_insert_common/benchmark_routing.py
```
