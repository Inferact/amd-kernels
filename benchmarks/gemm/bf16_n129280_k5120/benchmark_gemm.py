# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of bf16_gemm_n129280_k5120 against vLLM's rocm_unquantized_gemm_impl, A B B A: one graph of 100 calls
on 100 distinct activation sets, one weight copy (1.32 GB > the 256 MB MALL: HBM on every call, as in serving), 5
replays per round, median of 5 rounds.
Run: python3 benchmarks/gemm/bf16_n129280_k5120/benchmark_gemm.py [M ...]
"""
import statistics
import sys

import torch

from amd_kernels.gemm.bf16_n129280_k5120 import bf16_gemm_n129280_k5120, shuffle_weight, warmup_bf16_gemm_n129280_k5120
from amd_kernels.gemm.bf16_n129280_k5120.reference import K, N

ROUNDS, REPLAYS = 5, 5


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


def replay_us(graph, n):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(REPLAYS):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (REPLAYS * n)


def abba(ga, gb, n):
    ta, tb = [], []
    for _ in range(ROUNDS):
        for g, t in ((ga, ta), (gb, tb), (gb, tb), (ga, ta)):
            t.append(replay_us(g, n))
    return statistics.median(ta), statistics.median(tb)


if __name__ == "__main__":
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

    ms = [int(v) for v in sys.argv[1:]] or [16, 32, 64, 96, 128, 192, 256, 384, 512, 768]
    dev = torch.device("cuda")
    torch.manual_seed(441)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    ws = shuffle_weight(w)
    warmup_bf16_gemm_n129280_k5120(dev, True)
    print("M: kernel us (TB/s) | vLLM us | speedup")
    for m in ms:
        xs = [torch.randn(m, K, device=dev).to(torch.bfloat16) for _ in range(100)]
        out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        gk = capture([lambda x=x: bf16_gemm_n129280_k5120(x, ws, out, shuffled=True) for x in xs])
        gb = capture([lambda x=x: rocm_unquantized_gemm_impl(x, w) for x in xs])
        k, b = abba(gk, gb, len(xs))
        print(f"M {m:4d}: {k:7.1f} us ({N * K * 2 / k / 1e6:.2f} TB/s) | {b:7.1f} us | {b / k:.3f}x", flush=True)
