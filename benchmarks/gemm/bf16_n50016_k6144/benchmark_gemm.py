# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of bf16_gemm_n50016_k6144 for both weight layouts: one graph of 100 calls on 100 distinct activation
sets, one weight copy (615 MB > the 256 MB MALL: HBM on every call, as in serving), 10 replays x 5 rounds.
Run: python3 benchmarks/gemm/bf16_n50016_k6144/benchmark_gemm.py [M ...]
"""
import statistics
import sys

import torch

from amd_kernels.gemm.bf16_n50016_k6144 import bf16_gemm_n50016_k6144, shuffle_weight, warmup_bf16_gemm_n50016_k6144
from amd_kernels.gemm.bf16_n50016_k6144.reference import K, N


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
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / (10 * len(calls)))
    return statistics.median(times), min(times), max(times)


if __name__ == "__main__":
    ms = [int(v) for v in sys.argv[1:]] or [1, 16, 64, 70, 80, 112, 128, 256, 512, 1024, 2048]
    dev = torch.device("cuda")
    torch.manual_seed(441)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    ws = shuffle_weight(w)
    for shuffled in (False, True):
        warmup_bf16_gemm_n50016_k6144(dev, shuffled)
    for m in ms:
        xs = [torch.randn(m, K, device=dev).to(torch.bfloat16) for _ in range(100)]
        out = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        row = []
        for shuffled in (False, True):
            wk = ws if shuffled else w
            med, lo, hi = timing([lambda x=x, wk=wk, s=shuffled: bf16_gemm_n50016_k6144(x, wk, out, shuffled=s)
                                  for x in xs])
            row.append(f"{'shuf' if shuffled else 'row'} {med:.1f} [{lo:.1f}-{hi:.1f}] us "
                       f"({N * K * 2 / med / 1e6:.2f} TB/s)")
        print(f"M {m:5d}: " + "; ".join(row), flush=True)
