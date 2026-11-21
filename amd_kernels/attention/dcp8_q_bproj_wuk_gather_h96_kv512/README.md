# DCP8 DSA query path: q_b + W_UK + all-gather over the eight head shards

## What it does

The decode query of one DSA layer at TP8 / DCP8 (12 heads per rank, kv_lora 512, NoPE, q_pe 64 wide): `q_a` bf16
[M, 1536] (the RMSNorm'd q_c, identical on all ranks) -> q_b projection [1536 -> 12 x 192] -> per head `q_nope` [M, 128]
x W_UK [128, 512] -> the latent query bf16 [M, 96, 576] on every rank (head 12 r + h from rank r: `[latent 512 | q_pe
64]`), the tensor `triton_sparse_mla_fwd` reads. Stock vLLM runs five launches and an all-gather (skinny GEMM, `torch.bmm`,
`torch.cat`, AITER IPC all-gather of 110 KB per token). Both variants keep its rounding (fp32 accumulate, bf16 after
q_b, bf16 after W_UK; `reference.py`: cosine >= 0.9999, rel. L2 <= 4e-3, bitwise repeatable, no atomic adds into outputs):
- `q` (one launch, the decode path): this rank's q_b tiles (weights stationary in registers, 4.5 KB per token and peer) are pushed
  into both data areas of all eight ranks by epoch parity as contiguous per-tile runs; every rank applies W_UK to all 96
  heads from a replicated packed weight (12.6 MB per layer) into the caller's `out`, tile by tile as the tiles arrive.
  Optionally (`out_q8`) it writes the hi/lo fp8 record of the sparse-MLA decode kernel (`reference.quantize_q`,
  uint8 [M, 96, 1168]) instead of or next to the bf16 query, bit-exact against `quantize_q` of its own bf16 output.
- `latent`: q_b and the 12 local heads' W_UK, then the latent query is pushed (13.8 KB per token and peer): two
  launches, the result a view of the communicator's buffer; it moves 3x the bytes of `q`.
Weights are packed once at load (`pack_q_b`, `pack_w_uk`: MFMA operand tiles).

## Entry points

- `QueryGatherComm`: the eight-rank communicator (IPC gather buffer, flag words, device-side epoch); created
  collectively before capture, one per stream, closed on all ranks after graph work completes.
- `q_b_allgather_wuk_all_heads`: the `q` variant, graph-safe; what a decode layer calls in place of q_b, W_UK, concat,
  gather; returns `out`, `out_q8` or both. The dispatch is `host.config_q` (row-count tiers).
- `q_b_wuk_allgather_latent`: the `latent` variant, graph-safe; the result is valid until the next call.
- `pack_q_b`, `pack_w_uk`: load-time packing.

## Performance

Eight MI355X, graph replay of 16 calls over rotated weights, min of 12 rounds of the slowest rank, us per call. Stock =
q_b GEMM + `torch.bmm` + `torch.cat` + AITER all-gather in the same harness. Floor = launch gap + first q_b weights + the
push at the 343 GB/s all-to-all egress rate + last ack + flag + the last tile at the consumer, from ceilings measured
on MI355X. `gather` = the call with consumers that only wait; `P0` = the push and flag protocol alone (no
weight loads, no math): the floor of this protocol, 88-113 % of the model.

| M | stock | q | q vs stock | q8 | both | gather | P0 | floor | % of floor |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 24.1 | 10.3 | 2.34x | 11.0 | 11.1 | 8.7 | 7.0 | 6.9 | 67% |
| 24 | 27.5 | 13.0 | 2.12x | 14.5 | 15.1 | 10.2 | 8.1 | 8.0 | 62% |
| 40 | 32.4 | 15.3 | 2.12x | 17.1 | 18.1 | 13.4 | 10.7 | 9.5 | 62% |
| 65 | 41.8 | 19.6 | 2.14x | 22.1 | 24.1 | 16.4 | 12.6 | 11.8 | 60% |
| 80 | 46.3 | 21.2 | 2.18x | 24.1 | 25.9 | 17.1 | 13.2 | 13.2 | 62% |
| 96 | 51.2 | 24.2 | 2.12x | 29.6 | 30.8 | 19.2 | 14.5 | 14.8 | 61% |
| 128 | 59.0 | 27.7 | 2.13x | 33.6 | 34.0 | 22.0 | 16.2 | 17.8 | 64% |
| 160 | 66.5 | 31.4 | 2.12x | 37.4 | 39.6 | 24.8 | 18.4 | 20.8 | 66% |
| 192 | 76.0 | 35.4 | 2.15x | 43.4 | 44.6 | 28.1 | 21.0 | 23.8 | 67% |

## Tests

- `python3 -m pytest tests/attention/dcp8_q_bproj_wuk_gather_h96_kv512 -q` (8 GPUs, 120 s per call: `-k contract`, `-k outputs`
  (bf16 / fp8 record), `test_compile.py`)
- `python3 benchmarks/attention/dcp8_q_bproj_wuk_gather_h96_kv512/benchmark_query_gather.py` (8 GPUs)
