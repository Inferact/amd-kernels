# bf16 owner-row all-to-all and all-gather of logits, 20480 columns per source, eight ranks

## What it does
`RowAllToAll` moves the vocabulary-sharded logits rows of Kimi-K3.1 TP8 to the rank that owns each request, straight
into the final `[owned rows, 163840]` layout (the batch-sharded sampler's input), in place of RCCL's `all_to_all_single`
+ permute; with `counts=None` every rank receives every row: the all-gather of `tensor_model_parallel_all_gather(logits)`
-> `[M, 163840]`. The senders write into the receivers' buffers (push) and the receiver only waits on eight flag words in
its own memory. Input: bf16 `[rows, 20480]`, rows <= 256, sorted stably by owner. Contract (`reference.py`): rank r
receives rows `[sum(counts[:r]), sum(counts[:r + 1]))` of every source, concatenated along the vocabulary, every bf16
bit kept (NaN payloads too). Protocol (IPC buffers and flag blocks, two-level ticket, device sequence counter, the two
alternating buffers): module docstring of `kernel.py`. The rows are valid until the next exchange. No atomic output adds.

## Entry points
- `RowAllToAll`: a collective; a receive buffer of `max_rows x 163840 x 2` B per rank (84 MB at 256 rows).
- `RowAllToAll.warmup`: a collective before capture; compiles both pushes and the wait.
- `RowAllToAll.__call__`: push of rows `[rows, 20480]` (owner-sorted with `counts`, a CUDA int32 `[8]` the same on every
  rank; or all-gather without) and wait; read the rows with `recv(n)`.
- `RowAllToAll.gemm_exchange`: the lm_head shard GEMM (`gemm/bf16_n20480_k7168`) of `x[row_indices]` and the exchange of its
  output in one call: the GEMM pushes its rows itself up to 96 rows (owner mode) / 16 rows (all-gather), above that it writes them
  locally and one contiguous push moves them. This is the entry the model calls.
- `RowAllToAll.send_args`, `RowAllToAll.wait`, `RowAllToAll.recv`: for a producer that pushes the rows itself, then the wait and
  the in-place view.
- `RowAllToAll.close`: a collective, after every graph that uses the communicator is retired.

All ranks call every exchange in the same order, empty receivers included; one communicator per stream.

## Performance

MI355X, eight ranks, `benchmark_exchange.py`: hidden rows replicated, owner-sorted rows (request r on rank r % 8, 4 rows per
request), the lm_head GEMM included in every column, graph replay of 6 distinct inputs over three weight copies (880 MB),
minimum of nine rounds, MAX over ranks, us per call. `vLLM AG` = `rocm_unquantized_gemm_impl` +
`tensor_model_parallel_all_gather` (vLLM's default); `vLLM A2A` = GEMM + `all_to_all_single` + permute (vLLM's batch-sharded
sampling); `this AG` / `this owner` = `gemm_exchange` to every rank / to the owners (rows read in place). Ceiling = the GEMM at
its HBM or MFMA ceiling (6.5 TB/s / 1.5 PF) + 2.4 us (wait launch gap 1.6 + one flag hop 0.8) + the busiest link's bytes at
49 GB/s (the measured 8-GPU all-to-all: 343 GB/s out over seven links), the push counted after the GEMM.

| M | 12 | 24 | 40 | 65 | 80 | 96 | 128 | 160 | 192 |
|---|---|---|---|---|---|---|---|---|---|
| vLLM AG | 88.5 | 102.1 | 136.5 | 174.8 | 192.4 | 206.6 | 255.6 | 307.2 | 351.5 |
| vLLM A2A | 77.3 | 72.5 | 86.8 | 95.5 | 99.6 | 92.2 | 107.2 | 122.3 | 130.3 |
| this AG | 61.9 | 75.3 | 90.5 | 114.4 | 128.5 | 144.1 | 174.1 | 208.4 | 238.9 |
| this owner | 53.1 | 54.1 | 59.0 | 63.5 | 68.3 | 72.6 | 80.9 | 91.8 | 97.8 |
| owner vs vLLM AG / A2A | 1.67x / 1.46x | 1.89x / 1.34x | 2.31x / 1.47x | 2.75x / 1.50x | 2.82x / 1.46x | 2.85x / 1.27x | 3.16x / 1.33x | 3.35x / 1.33x | 3.59x / 1.33x |
| owner ceiling | 50.9 | 51.0 | 54.3 | 55.2 | 57.8 | 57.8 | 61.2 | 64.6 | 68.1 |
| owner % of ceiling | 96 | 94 | 92 | 87 | 85 | 80 | 76 | 70 | 70 |
| AG % of ceiling | 93 | 90 | 90 | 89 | 89 | 89 | 89 | 87 | 87 |

The GEMM's own push loses to a local write and one contiguous push above 96 rows (owner) / 16 rows (all-gather): the tile's 320-byte
row segments reach the receivers' HBM at a fraction of the rate of full-row streams (store-only pushes at 192 rows: 160 B segments 68 us,
320 B 54, 640 B 49, 1280 B 43, this entry's 20 KB streams 35). Standalone push + wait: 13.3 / 18.6 / 23.7 / 32.8 us at M 12 / 80 / 128 / 192
against link floors of 4 / 11 / 16 / 22.5 us at 49 GB/s: 80 % of the floor at 192 rows (below 4 MB per peer the xGMI ramp, not the
link, binds). The push cannot overlap the GEMM: every output tile completes with the last K block, and splitting the rows into
two passes re-reads the weights (+42 us).

## Tests
```bash
python3 -m pytest tests/collectives/all_to_all_bf16_n20480_ranks8 -q               # 8 GPUs
python3 benchmarks/collectives/all_to_all_bf16_n20480_ranks8/benchmark_exchange.py # 8 GPUs
```
