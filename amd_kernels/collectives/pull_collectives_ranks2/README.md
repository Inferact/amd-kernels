# Pull collectives over IPC-mapped pools, two ranks

## What it does

The SP-prefill collectives of MiniMax-M3.1-Flash TP2 in FlyDSL instead of RCCL, over the one xGMI link of a GPU pair:
the all-to-all with the MXFP8 unpack into o_proj_full's input (4096 code columns per rank), all-gathers of row-major
fields, the reduce-scatter, and the SP norm tails fused into them (callers: the two-rank `seq_parallel_handoffs`, MoE
`gather_mxfp4_moe` of `moe/mxfp4_h6144_i1536_e128_top4`). Shard rows R = T / 2, T = 4096..32768 tokens. IPC pools, two
slots and uncached ticket words per rank; stagers and pullers share one launch and hand off part by part. Bit for bit
(`reference.py`): gathers in rank order; `unpack_reference`; reduce-scatter `bf16((+0 + b0) + b1)` in fp32; slot
combine `(((s0+s1)+s2)+s3)+shared` in fp32, shared `[2R, H]` on every row or `[R, H]` (owner form) on the own rows;
norms per `norm/gemma_add_rmsnorm_h6144` (quant of the fp32 norm). The MoE tail's peer sums cross the link in
`lossless_x3.py`'s exact 12-bit code (9600 B per row instead of 12288).

## Entry points

All are built collectively (gloo, same sizes on both ranks), warmed up collectively on the current stream before
capture, one per ordered stream or DBO microbatch, called in the same order on both ranks; pools live for the process;
a communicator's grid is fixed for its lifetime. `Pools`, `Collectives` and `FencedAllGatherComm` keep their sequence
on the host (eager only); the others keep device counters, so eager calls and graph replays interleave.

- `amd_kernels.collectives.pull_collectives_ranks2.Pools`: a rank's IPC pools and flag words, one set per microbatch.
- `...Collectives`: one phase's slot of a set, bound to its stream; `alloc` gives a slot view the producer (the pack)
  writes and the all-to-all reads in place.
- `...FencedAllGatherComm.all_to_all_unpack_shuffled`: the attention rows' all-to-all into o_proj_full's MXFP8 codes
  and scales in `mxfp8_gemm`'s preshuffled layout.
- `...AllGatherComm.all_gather`: gathers contiguous 16 B-aligned row-major fields in rank order (the routed or compact
  MoE payload, codes + scales, bf16 rows).
- `...SlotsReduceScatterComm.slots4_reduce_scatter_add_rmsnorm_mxfp8`: the MoE tail (slot combine, shared add of
  either form, picked by its rows, reduce-scatter, residual add, norm, quant) into codes, shuffled scales, residual;
  `slots4_reduce_scatter` and `reduce_scatter` stop after the sum; the SDMA hand-offs' coded tail is
  `slots4_host._stage_rows_x3` + push + `_sum_norm_x3`.
- `...NormAllGatherComm.norm_all_gather`: the dense norm + quant fused into the gather (codes, scales, residual).

Each communicator's host file picks its kernels (`slots4_host.py`, `gather_host.py`, `all_gather_fence_h6144.py`,
`norm_gather_h6144.py`).

## Performance

Graph replay (the all-to-all eager), rank max, us per call at T tokens: ours / RCCL at world 2 (the same bytes, the
non-collective steps as separate launches); in brackets the fraction of the link roofline (R x row bytes at 61.2 GB/s,
the measured rate of one xGMI link with both directions busy; CU pulls top out at 57.4 GB/s, 93.8%). Bytes per row:
MoE tail 9600 (coded, owner-form shared; every-row shared within 0.5%), RS 12288, MoE gather 3296 (routed; the 7328 B
compact one: 276.2 / 538.7 / 1053.8 / 2101.7 us), norm gather 6336, a2a 4224.

| entry | T = 4096 | 8192 | 16384 | 32768 |
|---|---:|---:|---:|---:|
| MoE tail | 373.3 / 594.8 (86.1%) | 733.5 / 1147.7 (87.6%) | 1451.8 / 2209.4 (88.5%) | 2889.7 / 4305.7 (88.9%) |
| RS | 451.8 / 545.2 (91.0%) | 892.7 / 1061.4 (92.1%) | 1776.6 / 2072.3 (92.6%) | 3536.5 / 4040.1 (93.0%) |
| MoE gather | 128.1 / 143.5 (86.1%) | 250.4 / 276.4 (88.1%) | 480.9 / 518.1 (91.7%) | 951.3 / 1008.5 (92.8%) |
| norm gather | 238.1 / 296.4 (89.0%) | 465.6 / 555.0 (91.1%) | 923.9 / 1076.2 (91.8%) | 1834.2 / 2119.9 (92.5%) |
| a2a unpack | 162.7 / 240.2 (86.9%) | 314.8 / 420.8 (89.8%) | 616.3 / 791.4 (91.7%) | 1224.8 / 1535.7 (92.3%) |

## Tests

```bash
python3 -m pytest tests/collectives/pull_collectives_ranks2 -q                     # 2 GPUs
python3 benchmarks/collectives/pull_collectives_ranks2/benchmark_collectives.py r.json
python3 benchmarks/collectives/pull_collectives_ranks2/benchmark_link.py
```
