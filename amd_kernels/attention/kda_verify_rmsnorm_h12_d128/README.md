# KDA verification with compact recovery records and gated RMSNorm

## Computation

One gfx950 workgroup owns the FP32 recurrent checkpoint for one request and head.
The kernel reads post-convolution BF16 QKV, the projected raw gate, beta and output gate.
It evaluates the delta rule in FP32, rounds the attention result to BF16, and applies sigmoid-gated RMSNorm.
The output and the FP32 correction and BF16 raw key/gate records are written in the same launch.
The sampling step can use the records to commit any accepted prefix. The checkpoint is read-only during verification.

The entry serves 12 local heads of dimension 128, a negative safe-gate lower bound and windows of 1–8 tokens.
Offsets start at zero and delimit ordered spans of at most one window within the activation rows.
Each positive slot indexes a checkpoint. Nonpositive slots produce zero output for their tokens.
Empty requests, unused positions in the record buffers and rows beyond the last offset receive no writes.
Record rows follow request order. The checkpoint slot pitch can include page padding.

## Entry points

- `attention.kda_common.kda_verify_rmsnorm` fills the caller's output and recovery buffers.
- `attention.kda_common.kda_verify_rmsnorm_unsupported_reason` checks tensor metadata before dispatch.
- `attention.kda_common.kda_verify_rmsnorm_shapes` reports the supported head shapes.

## Performance

MI355X graph replay, median of nine interleaved rounds in µs per layer call. The baseline runs Triton verify,
gated RMSNorm and the output copy. Both consume the same post-convolution inputs and emit the same record format.
The rotating measurement cycles through at least 512 MiB of checkpoint slots. Other operands repeat.

| Window | Requests | Repeated slots, baseline / fused | Rotating slots, baseline / fused |
|---:|---:|---:|---:|
| 4 | 1 | 9.92 / 5.69 | — |
| 4 | 8 | 12.09 / 5.76 | 13.78 / 7.76 |
| 4 | 32 | 20.33 / 8.66 | 22.95 / 12.54 |
| 4 | 96 | 46.50 / 22.93 | 49.06 / 25.95 |
| 5 | 1 | 11.39 / 6.51 | — |
| 5 | 8 | 13.87 / 6.54 | 15.77 / 8.53 |
| 5 | 32 | 29.33 / 10.22 | 31.21 / 13.99 |
| 5 | 96 | 69.56 / 25.85 | 71.41 / 28.81 |

## Tests

```sh
python -m pytest tests/attention/kda_verify_rmsnorm_h12_d128 -q
python benchmarks/attention/kda_verify_rmsnorm_h12_d128/benchmark_verify.py --window 5
# With a vLLM runtime that exports RecoverSSM verify and gated RMSNorm
python benchmarks/attention/kda_verify_rmsnorm_h12_d128/benchmark_verify.py --compare-vllm --checkpoint-mib 512 --batches 8 32 96
```
