# KDA spec-verify decode: f_b gate, conv, delta rule, gated norm in one launch (12 heads, head dim 128)

## What it does

One launch for a Kimi Delta Attention (KDA) layer's spec-verify decode on one rank (12 heads of dim 128, TP8): the f_b
gate GEMM, the causal conv (width 4) + SiLU of q / k / v, the delta-rule recurrence over each sequence's tokens with
per-token state snapshots, and the sigmoid-gated RMSNorm. It replaces `kimi_k3_kda_fb_proj` + `causal_conv1d_update` +
`fused_recurrent_kda` + `FusedRMSNormGated` (or the AITER `kda_fused_recurrent_spec_decode` pair f_b GEMM + decode).

Inputs: the fused in_proj output `proj` [M, 6288] bf16 (q | k | v | gate | f_a | beta | pad, row stride a multiple of
8), `f_b_weight` [1536, 128] bf16, `A_log` [12] and `dt_bias` [1536] fp32, `conv_weight` [3, 4, 1536] fp32, the conv
state [slots, 4608, len] bf16 in the served (state_len, dim) layout (dim innermost), the recurrent state [slots, 12,
128, 128] fp32 (any slot pitch), `cu_seqlens` / `state_indices` [N, T] / `num_accepted` [N] int32, the output norm
weight (bf16 or fp32). It updates the conv and recurrent states in place and writes `out` [M, 12, 128] bf16. Token t of a
sequence reads its state from slot `state_indices[n, num_accepted - 1]` and writes the snapshot after token t to slot
`state_indices[n, t]` (slot <= 0 writes nothing; a sequence whose seed slot is <= 0 writes zeros).

Numerics (`reference.py` is the contract): fp32 throughout; g1 = bf16(f_a @ f_b^T) rounded once as the stock GEMM's
output; gate, conv + SiLU, L2 norms, recurrence and norm follow the AITER spec-decode kernel (cosine >= 0.9999 and
states within 2e-3 of the reference in the tests, identical conv state, two runs bit for bit equal).

## Entry points

- `kda_decode_fb_gate_spec`: the KDA core of one layer in a spec-verify step (an `amd_kernels.warm_up` entry).
- `blocker`: why a call does not fit (None: it fits); the caller falls back to the stock chain otherwise.

`host.py` compiles one kernel per mode (tokens per sequence, norm weight dtype, snapshot mode, uniform lengths); the
grid is one workgroup per (sequence, head).

## Performance

Graph replay, 1 MI355X, inputs > 700 MB (cold), median of 15 rounds, us per layer call. Skeleton: the same grid, table
lookups, state read and snapshot writes, no math. Ceiling: 6.2 TB/s (measured blend of 6.5 read / 6.0 write).

| M (rows x tok) | MB | fused us | TB/s | % ceiling | skeleton us | fused/skeleton | stock f_b + AITER us | speedup |
|---|---|---|---|---|---|---|---|---|
| 12 (2 x 6) | 11.8 | 9.8 | 1.21 | 20 | 5.6 | 1.73 | 16.2 | 1.66x |
| 24 (6 x 4) | 24.9 | 8.5 | 2.92 | 47 | 6.1 | 1.41 | 15.0 | 1.75x |
| 40 (10 x 4) | 41.2 | 9.6 | 4.29 | 69 | 7.0 | 1.36 | 17.5 | 1.82x |
| 65 (13 x 5) | 63.9 | 13.2 | 4.86 | 78 | 10.5 | 1.25 | 21.9 | 1.66x |
| 80 (20 x 4) | 81.9 | 15.2 | 5.38 | 87 | 12.7 | 1.20 | 22.1 | 1.45x |
| 128 (32 x 4) | 130.9 | 25.2 | 5.20 | 84 | 23.3 | 1.08 | 34.0 | 1.35x |
| 192 (48 x 4) | 196.1 | 34.7 | 5.66 | 91 | 31.0 | 1.12 | 48.2 | 1.39x |

Small M is a latency chain, not bandwidth (without the snapshot stores still 9.2 us at M=12): launch 1.6 us + loads 2.2
+ gate / conv 1.3 + VALU-bound token loop ~0.45 us a token + tail 1.3. M >= 65 is within 10-15% of launch + bytes / 6.2 TB/s.

## Tests

```bash
python3 -m pytest tests/attention/kda_decode_fb_gate_spec_h12_d128 -q
python3 benchmarks/attention/kda_decode_fb_gate_spec_h12_d128/benchmark_decode.py
```
