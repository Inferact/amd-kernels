# EP8 MXFP8 × MXFP4 routed experts with SiTUv2

## Computation

Each of eight gfx950 ranks owns 112 full experts. One persistent launch per rank dispatches distinct top-16 routes, computes gate/up, SiTUv2 (4, 25), intermediate MXFP8 quantization and down projection, then returns weighted BF16 slots to their source and sums them in slot order in FP32. The result is BF16. Input width is 3584 and expert intermediate width is 3072. Each rank supplies 0–256 BF16 or prequantized MXFP8 rows. Out-of-range expert IDs contribute zero.

## Entry points

- `amd_kernels.moe.fused_ep_expert_weights` prepares a layer's shuffled weight and scale views.
- `amd_kernels.moe.expert_parallel` creates the rank's collective workspace. Call `warm_up` on all ranks before capture.
- The shape dispatch lives in `amd_kernels/moe/__init__.py`. `reference.py` specifies the numerics.

## Performance

Eight MI355X GPUs, BF16 input, maximum-rank graph latency in microseconds. Each replay rotates four synthetic batches. The baseline and selected tile used identical inputs in each pair.

| Rows per rank | Uniform, fixed BM16 | Uniform, selected | All routes to one rank, fixed BM16 | All routes to one rank, selected |
|---:|---:|---:|---:|---:|
| 8 | 242.87 | 239.94 | 334.67 | 311.40 |
| 32 | 327.80 | 319.43 | 894.93 | 499.60 |
| 96 | 444.30 | 343.75 | 2316.41 | 963.31 |
| 256 | 977.40 | 398.78 | 5899.00 | 1559.87 |

## Tests and benchmark

```sh
python -m pytest tests/moe/fused_ep_mxfp4_h3584_i3072_e112_top16_ranks8 -q
python -m torch.distributed.run --standalone --nproc-per-node=8 benchmarks/moe/fused_ep_mxfp4_h3584_i3072_e112_top16_ranks8/benchmark_experts.py
```
