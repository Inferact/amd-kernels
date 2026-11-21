# Dense MLA decode, 96 latent heads, e4m3 latent KV

## What it does

The decode attention of the DSpark draft layers of Kimi-K3.1 (5 dense MLA layers, 96 heads, latent 512 + RoPE 64) over the
draft's own e4m3 KV cache under DCP8: this rank holds every 8th token of each context, the 12-head queries of the 8 ranks are
gathered to 96, and each request carries a causal block of 3 or 4 query rows (k of the served configs c16, c24: 4; c32, c48: 3).
`q [requests * rows, 96, 576]` e4m3 (scale `q_scale`), the cache a flat e4m3 `[N, 576]` (scale `kv_scale`; the first 512 bytes
are also V), one int32 key list per request (`kv_indptr`, `kv_indices`: flat cache rows, any layout) and the global lengths
`g_kv_indptr`. Row t of a request sees the local keys j with `j * 8 + rank <= G - rows + t`. Output: bf16 `[rows, 96, 512]` and
the natural-log LSE `[rows, 96]` fp32 (what the DCP merge needs; a row with no visible key: 0 and -inf).

Contract: `reference.py` (fp64 softmax on the e4m3 inputs). The kernel rounds P to e4m3 before the PV product, as
AITER's `mla_a8w8` does; both land at cosine 0.9995 against the fp64 result on random data (the e4m3 rounding of P), so
the kernel's cosine against the reference is >= 0.998 in the tests and against AITER ~0.999; two runs match bit for bit.
Exact semantics (round-robin causal window, scales, LSE) match `mla_decode_fwd(..., cp_world_size=8, g_kv_indptr=...)`.

## Entry points

- `fp8_dense_mla_decode`: the attention of one draft layer (partials + merge, two launches). An `amd_kernels.warm_up` entry:
  the first call of a (rows, cp world) mode compiles both kernels. The request count is a run-time argument (at most 63) and
  every request length too; CUDA-graph padding requests have empty key lists.
- `supports_fp8_dense_mla_decode`: whether a call fits (gfx950, e4m3 q and cache, int32 CSR, float32 device scales).

`host.py` sets the split (85 key chunks, 255 workgroups); `kernel.py` is the partials, `merge.py` the merge of a request's
partials.

## Performance

MI355X, 15000 local keys per request, min of 15 graph replays, us per call. Stock: AITER's `mla_decode_fwd` as vLLM runs
it (`benchmark_decode.py`).

| case | reqs x rows | stock us | kernel us | speedup | % of 4.07 PF MFMA |
|---|---|---|---|---|---|
| c16 | 8 x 4 | 104.8 | 63.8 | 1.64x | 38.6 |
| c24 | 13 x 4 | 249.3 | 90.4 | 2.76x | 44.3 |
| c32 | 20 x 3 | 179.6 | 132.6 | 1.35x | 34.8 |
| c48 | 48 x 3 | 422.9 | 286.6 | 1.48x | 38.7 |

## Tests

```bash
python3 -m pytest tests/attention/fp8_dense_mla_decode_h96_kv512 -q
python3 benchmarks/attention/fp8_dense_mla_decode_h96_kv512/benchmark_decode.py --check
```
