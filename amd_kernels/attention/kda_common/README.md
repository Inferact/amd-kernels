# KDA entries: spec-verify decode, chunked prefill and mono decode, routed by shape

## What it does

One entry per Kimi Delta Attention (KDA) op. A caller passes tensors; the entry reads the shape from them and calls
the shape package that serves it, so the caller imports no shape package:

| op | key, read from | shape packages |
|---|---|---|
| `kda_decode_fb_gate_spec` | (heads, head_dim) of the recurrent state [slots, heads, d, d] | `kda_decode_fb_gate_spec_h12_d128` |
| `kda_prefill` | (heads, head_dim) of the recurrent state cache | `kda_prefill_h12_d128` |
| `kda_verify_rmsnorm` | (heads, value dim) of the read-only checkpoint | `kda_verify_rmsnorm_h12_d128` |
| `kda_mono_decode` | (hidden, heads, head_dim) of `prefix` [S, hidden] and the state | `kda_mono_decode_k1_h12_d128` |

The single-rank kernels take no world size: 12 local heads already are the TP8 shard. Arguments, dtypes and the
numerics contract are the routed package's (its `reference.py`); the entry's outputs and in-place states are bit for bit
those of the direct package call. For a call no package serves, `*_unsupported_reason` (the op's own arguments) returns
a one-line reason and the caller falls back; it reads only shapes, dtypes, strides and host ints (no device read, safe
inside CUDA graph capture). The op itself only looks its key up and raises `ValueError` with the same "no kernel for"
text off the table.

## Entry points

- `kda_common.available`: gfx950 with FlyDSL importable (load-time check).
- `kda_common.kda_decode_fb_gate_spec`: the KDA core of one layer in a spec-verify step (f_b gate, conv, delta rule,
  gated norm); `kda_decode_fb_gate_spec_unsupported_reason`, `kda_decode_fb_gate_spec_shapes` (covered keys).
- `kda_common.kda_prefill`: the chunked prefill of one layer; `kda_prefill_unsupported_reason`, `kda_prefill_shapes`.
- `kda_common.kda_prefill_chunk_table`: the per-step chunk table `kda_prefill` takes, routed by `state_cache`.
- `kda_common.kda_mono_workspace`: the persistent decode's workspace for a (hidden, heads, head_dim), built before any
  activation exists (`begin_step()` once a step, `failed()`); `kda_mono_workspace_unsupported_reason`.
- `kda_common.kda_mono_decode`: one KDA layer from the AttnRes seam to the gated core output on that workspace;
  `kda_mono_decode_unsupported_reason` (includes "workspace built for <key>, call is <key>"), `kda_mono_decode_shapes`.

Dispatch: `_SHAPE_HOSTS` in `decode.py`, `prefill.py`, `mono.py` (key -> module name); a shape package is imported on
the first call of its key; `kda_mono_decode` routes by its workspace's package.

## Performance

The entry adds only the dispatch: a CUDA graph of entry calls holds the same kernels as one of direct calls. MI355X,
graph of 10 calls on the same (warm) inputs, min of 15 replays, us per call; host us per eager call with the routed
function replaced by a no-op (entry minus direct), and of the reason the caller runs.

| call | direct graph us | entry graph us | dispatch host us | reason host us |
|---|---|---|---|---|
| decode 12 rows (2 x 6) | 10.28 | 10.49 | 0.35 | 5.30 |
| decode 192 rows (32 x 6) | 32.10 | 31.95 | 0.34 | 5.55 |
| prefill 2048 tokens (one launch) | 81.03 | 80.46 | 0.65 | 5.95 |
| prefill 4096 tokens (staged) | 116.20 | 116.42 | 0.65 | 6.08 |
| mono 12 rows (2 x 6) | 29.44 | 28.86 | 0.25 | 5.75 |
| mono 16 rows (4 x 4) | 36.12 | 35.94 | 0.25 | 5.79 |

## Tests

```bash
HIP_VISIBLE_DEVICES= python3 -m pytest tests/attention/kda_common/test_routing.py -q   # host only
python3 -m pytest tests/attention/kda_common -q
python3 benchmarks/attention/kda_common/benchmark_routing.py
```
