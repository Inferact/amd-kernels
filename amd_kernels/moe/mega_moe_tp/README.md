# MegaMoE TP: the tensor-parallel MXFP4 MoE layer as one kernel (gfx950)

## What it does

One launch runs a TP rank's whole MoE layer: experts replicated over the group, `inter_dim` sharded; the collectives
around GEMM1 + activation (SiLU, SwiGLU-OAI, SiTUv2) + GEMM2 of this rank's inter slice. `comm_mode` `ag_rs`: each rank
passes its own tokens `[m, H]` bf16 and gets its rows of the summed output (sequence parallel); `ar`: replicated
`[M, H]` in, the all-reduced `[M, H]` out on every rank. Weights MXFP4 with e8m0 scales per 32 in `layout.py`'s
layouts; activations MXFP4, exchanged rows FP8 (`comm_dtype="bf16"`: bf16). The kernel is persistent (one CTA per CU,
CTAs and ranks spin on each other) and needs the whole GPU; a wait times out after 2 s and sets a sticky error bit.
Optional fused tail (ag_rs): residual add + (Gemma)RMSNorm + FP8 rows. Big batches (beyond the small-batch engine's
capacity, e.g. 2048 local tokens on 256 CUs) run a chunked big-M kernel with SDMA transfers; it needs MORI built with
SDMA (importable, or `AKR_MORI_PATH`). Launch configs per batch: `mega_moe_tp_tuned.csv` /
`mega_moe_tp_bigm_tuned.csv` (`AKR_CONFIG_MEGA_MOE_TP`, `AKR_CONFIG_MEGA_MOE_TP_BIGM`). Contract (`reference.py`):
`mega_moe_tp_ref` summed over the ranks within rel. L2 0.1; graph replay = eager bit for bit; ar output identical on
every rank. SpRsNorm: SP int8 reduce-scatter + residual add + Gemma RMSNorm (+ sigmoid top-k router) in one launch.

## Entry points

- `MegaMoeTP`: one layer (`MegaMoeTPConfig`); `forward` is a collective (same call count, order and local token
  count on every rank); `prepare(local_tokens)` before serving and graph capture compiles, runs and checks every launch
  config (rule 4); `set_weights` reuses one instance across layers of one shape; `error_flag` / `check_errors` /
  `reset` / `close` for the watchdog and teardown (module docstring of `layer.py`).
- `shuffle_weight`, `shuffle_scale`: the checkpoint's row-major MXFP4 codes / e8m0 scales into the kernel layouts.
- `SpRsNorm`: the sequence-parallel tail of an MoE layer whose partials are not reduced by the kernel.
- `ActivationType`, `mega_moe_tp_supported`, `MegaMoeTPLDSError`, `ERR_BIGM`, `COMM_MODES`.

Dispatch: `mega_moe_tp.MegaMoeTPEngine.config` picks the schedule and row tile per batch (tuned CSV, else defaults);
`layer.MegaMoeTP.forward` sends big batches to `bigm_compute.MegaMoeTPBigM`.

## Performance

4 x MI355X, m3 (H 6144, I 768 per rank, 129 experts, top 5, SwiGLU), fp8 comm, random routing; graph replay of 10
calls over 4 inputs, median of 20 replays, slowest rank, us per call. No baseline measured here.

| M (global tokens) | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ag_rs | 66.3 | 98.4 | 136.0 | 163.4 | 209.1 | 243.6 | 258.3 | 264.8 | 412.3 | 611.7 | 1148.3 |
| ar | 63.5 | 96.2 | 132.0 | 164.0 | 193.4 | 211.1 | 239.8 | 320.4 | 490.1 | 825.7 | 1587.8 |

## Tests

```bash
cd amd-kernels && source env.sh   # or: PYTHONPATH=$PWD/.cache/flydsl-0.5.0:$PWD
python3 -m pytest tests/moe/mega_moe_tp -q                       # 4 GPUs; big-M cases need MORI
python3 benchmarks/moe/mega_moe_tp/benchmark_mega_moe_tp.py      # 4 GPUs
```
