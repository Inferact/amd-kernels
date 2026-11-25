# DCP8 LSE combine of packed partials (16 heads per rank, value dim 192)

## What it does

Decode with DCP8 on 8 GPUs: every rank holds the partial attention output of its KV shard for all 128 heads, already
packed per owner as vLLM's a2a merge sends it: `send [8, T, 48, 66]` bf16 (T <= 512), where `send[d, t]` holds the 16
heads of rank d as 48 sub-heads of 64 values (value dim 192 split in 3), each followed by its fp32 base-2 LSE as two
bf16 words (empty shards already -inf). One launch replaces RCCL `all_to_all_single` of that buffer plus the Triton
`_dcp_a2a_unpack_combine_kernel` and returns this rank's heads 16 d .. 16 d + 15 as `out [T, 16, 192]` bf16.
Each rank pushes `send[d]` over xGMI straight into rank d's slot (system-scope stores, one flag per 16-row tile and
source); the owner weighs the 8 sources by `2^(lse_s - g)`, `g = log2(sum 2^(lse_s - max)) + max`, and sums in source
order in fp32, rounded once. `reference.py` is the contract; the output is bit-exact against vLLM's a2a merge (the
same exp2 / log2 lowering, unfused multiply then add). No atomics; one compiled variant (T is a runtime argument).

## Entry points

- `DcpPackedCombine`: the communicator (a collective: uncached IPC pools, control blocks, peer table), one per stream,
  built before graph capture.
- `dcp8_lse_combine_packed`: the entry (`amd_kernels.warm_up`): one call per attention layer on every rank in the same
  order with the same T. The first call compiles the single variant.
The protocol (two slots, per-tile flags, start words and the counter bumped by the last workgroup, so CUDA graphs
replay with no host state) and the layout are in the module docstring of `kernel.py`.

## Performance

8 x MI355X, CUDA-graph replay, inputs rotating over > 300 MB, MAX over ranks, us per call (median / min of 9),
baseline: RCCL `all_to_all_single` + vLLM's `_dcp_a2a_unpack_combine_kernel` on the same inputs. Ceiling: launch 1.6 us
+ flag 0.8 us + bytes sent to the 7 peers / 343 GB/s + 0.6 us tail (constants of `dcp8_lse_combine_vup_mla_h12_kv512`).

| T | sent MB | ceiling us | stock us | fused us | speedup | % of ceiling |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.04 | 3.1 | 14.1 / 13.2 | 13.9 / 12.7 | 1.02x | 25% |
| 12 | 0.53 | 4.6 | 17.0 / 16.1 | 16.7 / 15.6 | 1.02x | 29% |
| 32 | 1.42 | 7.1 | 28.2 / 26.8 | 18.3 / 17.8 | 1.54x | 40% |
| 64 | 2.84 | 11.3 | 35.3 / 34.4 | 22.5 / 21.8 | 1.57x | 52% |
| 128 | 5.68 | 19.6 | 42.6 / 41.7 | 30.0 / 29.0 | 1.42x | 68% |
| 192 | 8.52 | 27.8 | 53.6 / 52.2 | 37.8 / 36.7 | 1.42x | 76% |
| 240 | 10.64 | 34.0 | 65.1 / 63.1 | 42.7 / 41.8 | 1.53x | 81% |
| 288 | 12.77 | 40.2 | 72.1 / 70.2 | 48.6 / 47.7 | 1.48x | 84% |
| 384 | 17.03 | 52.7 | 91.1 / 90.3 | 59.9 / 59.2 | 1.52x | 89% |
| 512 | 22.71 | 69.2 | 115.0 / 113.3 | 75.0 / 73.8 | 1.53x | 94% |

Speedup is median / median, the ceiling share is ceiling / min. Output vs the stock sequence: bit-equal on every rank at
every T. Below 32 rows both paths sit at the per-call cross-rank handshake (~13 us).

## Tests

```bash
python3 -m pytest tests/attention/dcp8_lse_combine_packed_h16_v192 -q                    # 8 GPUs (+ vLLM for bit-exact)
AKR_COMPILE_ONLY=1 python3 -m pytest tests/attention/dcp8_lse_combine_packed_h16_v192/test_compile.py -q   # 1 GPU
python3 benchmarks/attention/dcp8_lse_combine_packed_h16_v192/benchmark_combine.py                 # 8 GPUs
```
