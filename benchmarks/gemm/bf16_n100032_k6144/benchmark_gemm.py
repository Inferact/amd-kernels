# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of bf16_gemm_n100032_k6144 (shuf layout, contiguous and the publish body) against vLLM's lm_head GEMM
(``rocm_unquantized_gemm_impl`` on the row-major shard, VLLM_ROCM_USE_AITER=1): one graph of 50 calls on 50 distinct
activation sets, one weight copy (1229 MB > the 256 MB MALL: HBM on every call, as in serving), 10 replays x 5
alternated rounds, median us.
Run: python3 benchmarks/gemm/bf16_n100032_k6144/benchmark_gemm.py [M ...]
"""
import os
import statistics
import sys

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

import torch

from amd_kernels.gemm.bf16_n100032_k6144 import bf16_gemm_n100032_k6144, shuffle_weight, warmup_bf16_gemm_n100032_k6144
from amd_kernels.gemm.bf16_n100032_k6144.reference import K, N

CALLS = 50


def capture(calls):
    for f in calls[:2]:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def replay_us(graph):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (10 * CALLS)


def roofline_us(m):
    hbm = (N * K + m * K + m * N) * 2 / 6.5e12
    return max(hbm, 2 * m * N * K / 2.5e15) * 1e6


def vllm_gemm():
    try:
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl
        return rocm_unquantized_gemm_impl
    except Exception as e:  # noqa: BLE001
        print(f"no vLLM baseline: {e}")
        return None


if __name__ == "__main__":
    ms = [int(v) for v in sys.argv[1:]] or [1, 8, 16, 64, 80, 128, 192, 224, 256, 320, 384, 416, 448, 512, 832, 1024,
                                           2048]
    dev = torch.device("cuda")
    torch.manual_seed(441)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    ws = shuffle_weight(w)
    warmup_bf16_gemm_n100032_k6144(dev, True)
    warmup_bf16_gemm_n100032_k6144(dev, True, "publish")
    base = vllm_gemm()
    print("M | vLLM us | contiguous us | publish body us | roofline us | contiguous / roofline", flush=True)
    for m in ms:
        xs = [torch.randn(m, K, device=dev).to(torch.bfloat16) for _ in range(CALLS)]
        out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        idx = torch.randperm(m, device=dev)
        pool = torch.empty(2 * m * N * 2, dtype=torch.uint8, device=dev)
        counter = torch.zeros(1, dtype=torch.int32, device=dev)
        flag = torch.zeros(16, dtype=torch.int32, device=dev)
        publish = (pool.data_ptr(), m * N * 2, counter, flag.data_ptr())
        variants = {"contiguous": [lambda x=x: bf16_gemm_n100032_k6144(x, ws, out, shuffled=True) for x in xs],
                    "publish": [lambda x=x: bf16_gemm_n100032_k6144(x, ws, shuffled=True, row_indices=idx,
                                                                    publish=publish) for x in xs]}
        if base is not None:
            variants["vllm"] = [lambda x=x: base(x, w) for x in xs]
        graphs = {k: capture(v) for k, v in variants.items()}
        times = {k: [] for k in graphs}
        for _ in range(5):
            for k, g in graphs.items():
                times[k].append(replay_us(g))
        med = {k: statistics.median(v) for k, v in times.items()}
        roof = roofline_us(m)
        print(f"{m} | {med.get('vllm', float('nan')):.1f} | {med['contiguous']:.1f} | {med['publish']:.1f} | "
              f"{roof:.1f} | {roof / med['contiguous']:.2f}", flush=True)
        del graphs, xs, pool
        torch.cuda.empty_cache()
