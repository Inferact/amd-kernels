# Two-rank sampler-output gather, eight int64 token columns

## What it does

`SamplerAllGather` gathers the sampler output of MiniMax-M3.1-Flash TP2 (DSpark) across two ranks in one FlyDSL
workgroup: it packs each rank's tokens, sampled / rejected counts and per-request NaN sums, exchanges them, and unpacks
the requested rows in batch order, in place of the pack, `tensor_model_parallel_all_gather` and unpack of vLLM's
`vllm/v1/worker/gpu/sample/batch_shard.py::gather_sampler_output`; `_gather_logprobs_tensors` stays the caller's.
Contract (`reference.py`): integer-exact; tokens int64 with 1-8 input columns, zero-padded to 8 output columns;
counts int32; the NaN input is int32 per logit row, counted after masking, summed per request over `cu` (at most 8
logit rows per request); `src[i]` decodes as `(rank, local_row) = divmod(src[i], src_stride)`. Unused sampled-token
suffix bytes are copied as they are. No sampling arithmetic, no atomic add on any output.

## Entry points

- `SamplerAllGather`: a collective on the two ranks of a gloo group; a two-slot IPC pool (2 x 22528 bytes) and a
  256-byte uncached flag buffer per rank (IPC helpers of `all_to_all_bf16_n100032_ranks2`).
- `SamplerAllGather.warmup`: a collective before graph capture; compiles the one variant, which covers every size,
  column count and NaN mode.
- `SamplerAllGather.__call__`: gathers tokens, sampled / rejected counts and NaN sums into caller-supplied outputs in
  batch order (at most 256 local and 256 output rows). With NaN counting off every rank passes no NaN tensors; an
  empty sender passes zero-length tensors and a one-entry prefix. vLLM passes `metadata.gathered_src_indices` and
  `max_num_reqs_per_rank`.
- `SamplerAllGather.close`: a collective, after every graph that uses the communicator is retired.

Both ranks call in the same order; one communicator per stream. Shapes are runtime arguments fixed inside a captured
graph; payloads, prefix sums and maps may change on every replay. `I64C7AllGather` reuses this pool and protocol.

## Performance

Two GPUs of one MI355X node. Graphs of 100 calls on 100 distinct payloads, ten replays per round, five alternated
rounds, MAX over ranks, median us, dense owners, NaN counts on. Baseline: vLLM's Triton pack + RCCL all-gather +
Triton unpack, in the same benchmark.

| requests | 10 | 16 | 32 | 40 | 48 | 52 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 10.67 | 10.69 | 10.94 | 10.87 | 11.82 | 11.95 | 11.88 | 12.11 | 12.43 |
| this entry | 7.62 | 7.63 | 7.76 | 7.74 | 7.78 | 7.85 | 8.00 | 8.52 | 10.43 |

## Tests

```bash
python3 -m pytest tests/collectives/all_gather_sampler_i64_c8_ranks2 -q               # 2 GPUs
python3 benchmarks/collectives/all_gather_sampler_i64_c8_ranks2/benchmark_gather.py   # 2 GPUs
```

Every byte against `reference.py`, eager and replay: changed payloads and maps, empty senders / receivers, all rows on
one rank, narrow token rows, odd counts, 1000 slot uses, int32 sequence wrap, no compile after warm-up.
