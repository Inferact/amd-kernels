# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of bf16_latent_down_mxfp4_quant_k7168_n3584 against vLLM's rocm_unquantized_gemm_impl (VLLM_ROCM_USE_AITER=1)
followed by aiter's per_1x32_mx_quant_hip: one graph of 12 calls over 12 weight copies (12 x 51 MB > the 256 MB MALL: HBM on every
call, as in serving) and 12 activation sets, 10 replays x 15 rounds, min us, with the roofline of the weight stream.
Run: python3 benchmarks/gemm/bf16_latent_down_mxfp4_quant_k7168_n3584/benchmark_down_quant.py [M ...]
"""
import os
import sys

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")  # vLLM's tuned aiter hgemm path, as served

import torch

from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584 import (bf16_latent_down_mxfp4_quant_k7168_n3584,
                                                                       shuffle_weight)
from amd_kernels.gemm.bf16_latent_down_mxfp4_quant_k7168_n3584.reference import K, N

HBM_NT_TBPS = 6.95  # measured nontemporal HBM read
GRAPH_NODE_US = 1.57  # measured empty kernel in a graph
PEAK_BF16_TFLOPS = 1980.0  # measured register-resident MFMA
COPIES = 12


def timing(calls):
    for f in calls[:2]:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / (10 * len(calls)))
    return min(times)


if __name__ == "__main__":
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    ms = [int(v) for v in sys.argv[1:]] or [1, 12, 24, 40, 65, 80, 128, 192]
    torch.manual_seed(0)
    ws = [(torch.randn(N, K, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(COPIES)]
    wss = [shuffle_weight(w) for w in ws]
    wbytes = N * K * 2
    floor = GRAPH_NODE_US + wbytes / HBM_NT_TBPS / 1e6
    print(f"weight stream floor {floor:.1f} us")
    print(f"{'M':>4} {'stock gemm':>11} {'+ quant':>8} {'fused':>7} {'% of floor':>10} {'TB/s':>6} "
          f"{'TFLOPs':>7} {'% bf16 peak':>11} {'speedup':>8}")
    for m in ms:
        xs = [torch.randn(m, K, device="cuda").to(torch.bfloat16) for _ in range(COPIES)]
        gemm = timing([lambda x=x, w=w: rocm_unquantized_gemm_impl(x, w) for x, w in zip(xs, ws)])
        stock = timing([lambda x=x, w=w: per_1x32_mx_quant_hip(rocm_unquantized_gemm_impl(x, w),
                                                                quant_dtype=dtypes.fp4x2, shuffle=False)
                        for x, w in zip(xs, ws)])
        fused = timing([lambda x=x, w=w: bf16_latent_down_mxfp4_quant_k7168_n3584(x, w, shuffled=True)
                        for x, w in zip(xs, wss)])
        tbps = wbytes / fused / 1e6
        tflops = 2 * m * N * K / fused / 1e6
        print(f"{m:4d} {gemm:11.1f} {stock:8.1f} {fused:7.1f} {100 * floor / fused:10.0f} {tbps:6.2f} "
              f"{tflops:7.0f} {100 * tflops / PEAK_BF16_TFLOPS:11.1f} {stock / fused:7.2f}x", flush=True)
