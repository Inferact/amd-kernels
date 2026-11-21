# DSA / MLA decode attention: one entry per op (gfx950)

## What it does

The entries a caller imports instead of the shape packages of the DSA / MLA decode attention (prologue norms + KV
insert, indexer query prep, DCP query gather, sparse MLA decode, LSE combine + W_UV, dense draft MLA decode). Each op
reads its key from its operands (or, for an 8-rank op, from the communicator it was built with), imports the shape
package serving that key on the first call and passes the arguments through unchanged: outputs, numerics and
determinism are the shape package's (its `reference.py`). The 8-rank communicators are built by factories that take
the key dims as keywords and the world size from the group. Off the covered keys an op raises `ValueError` with the
"no kernel for" reason; `<op>_unsupported_reason` returns that reason, or any other one-line reason the package would
reject the call for (layout, dtype, row count, communicator), or None, so a caller can fall back instead. Reasons are
host-only (no device read, no allocation) and safe inside graph capture.

## Entry points

- `available`: gfx950 with FlyDSL, the device every routed package needs.
- `qkv_a_norm_kv_insert`: the q_a / kv_a RMSNorms and the fp8_ds_mla KV insert of a layer; `index` (a dict of the
  package's `IndexSide` fields) adds the indexer's key insert and head weights.
- `indexer_prep`: the indexer query's MXFP4 quant and top-k reset (with the key: its insert and the head weights).
- `q_gather_comm`: the query gather's 8-rank communicator (a collective, built before capture, `.close()` after).
- `pack_q_b`, `pack_w_uk`: load-time packing of q_b and W_UK for `q_bproj_wuk_gather`.
- `q_bproj_wuk_gather`: q_b + W_UK + all-gather: the latent query of all heads on every rank (bf16 and / or q8).
- `fp8_sparse_mla_decode`: a DSA layer's sparse MLA decode at DCP, `(out, lse)`.
- `lse_combine_comm`, `prepare_w_uv`: the combine's 8-rank communicator and its load-time W_UV layout.
- `lse_combine_vup`: the DCP LSE combine + W_UV up-projection (+ output gate) of this rank's heads.
- `fp8_dense_mla_decode`: the draft layers' dense fp8 MLA decode over a round-robin context shard, `(out, lse)`.
- `<op>_unsupported_reason` (the op's arguments), `<factory>_unsupported_reason`: None or why the call is not served.
- `<op>_shapes`: the covered keys, for a decision made before any operand exists (layer load, warm-up plan).
- `<op>_kernel` (all ops but the dense decode): the routed `amd_kernels.warm_up` entry, to compile without launching.

Dispatch: `_SHAPE_HOSTS` of each module (`prologue.py`, `query.py`, `sparse.py`, `combine.py`, `dense.py`).

## Performance

The entry adds only the dispatch: a graph holds the routed package's launches, nothing else. Graph replay of 20
calls (same operands), us per call, min of 30 replays alternating entry / direct; one MI355X, 8 for the 8-rank ops (max
over ranks). Host: Python the entry adds per eager call (the routed function replaced by a no-op) and one
`<op>_unsupported_reason` call, min of 15 x 2000 calls.

| op, shape | direct us | entry us | added us | host: dispatch added us | host: reason us |
|---|---|---|---|---|---|
| qkv_a_norm_kv_insert + index, M 12 | 2.44 | 2.43 | -0.01 | 0.56 | 5.60 |
| qkv_a_norm_kv_insert + index, M 192 | 2.64 | 2.58 | -0.06 | 0.55 | 5.61 |
| indexer_prep, M 12 / 192 | 2.38 / 2.49 | 2.38 / 2.48 | 0.00 / -0.01 | 0.32 | 2.91 |
| fp8_sparse_mla_decode, M 12 / 192 | 9.41 / 29.53 | 9.34 / 29.57 | -0.06 / +0.04 | 0.37 | 4.50 |
| fp8_dense_mla_decode, 13 x 4 / 48 x 3 rows | 33.73 / 58.22 | 33.77 / 58.18 | +0.04 / -0.04 | 0.28 | 4.42 |
| q_bproj_wuk_gather (8 ranks), M 12 / 192 | 9.37 / 35.41 | 9.42 / 35.43 | +0.05 / +0.02 | 0.11 | 1.73 |
| lse_combine_vup (8 ranks), M 12 / 192 | 9.25 / 24.34 | 9.21 / 24.27 | -0.04 / -0.07 | 0.09 | 2.06 |

## Tests

```bash
python3 -m pytest tests/attention/mla_decode_common -q   # the *_ranks8.py files need 8 GPUs
python3 benchmarks/attention/mla_decode_common/benchmark_routing.py [--ranks8]
```
