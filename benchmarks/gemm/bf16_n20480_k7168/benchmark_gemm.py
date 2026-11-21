# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""lm_head shard GEMM [M, 7168] x [20480, 7168]^T vs the served stock GEMM and hipBLASLt, with the measured ceilings.

Graph replay of 6 calls over three weight copies (880 MB, larger than the LLC) and distinct inputs; the minimum of five
rounds of ten replays, us per call (the GPU may be shared). Stock = AITER's `tgemm.mm` (what vLLM's
`rocm_unquantized_gemm_impl` calls with VLLM_ROCM_USE_AITER=1: the FlyDSL band-stream kernel up to 128 rows,
hipBLASLt above). Ceiling = max(weight bytes / 6.5 TB/s (measured HBM read ceiling), 2 M N K / 1.5 PFLOP/s (measured
library bf16 GEMM ceiling; the MFMA peak is 1.98 PFLOP/s)); `% ceiling` = ceiling / this entry.
"""
import argparse
import os

import torch

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
from aiter.tuned_gemm import tgemm  # noqa: E402

from amd_kernels.gemm.bf16_n20480_k7168 import bf16_gemm_n20480_k7168  # noqa: E402
from amd_kernels.gemm.bf16_n20480_k7168.reference import K, N  # noqa: E402

HBM_TBS, PEAK_TFLOPS = 6.5, 1500.0


def replay_us(fn, n):
    graph = torch.cuda.CUDAGraph()
    fn(0)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for i in range(n):
            fn(i)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(15):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(10):
            graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (10 * n))
    return min(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ms", type=int, nargs="*", default=[1, 4, 8, 12, 16, 24, 40, 65, 80, 96, 128, 160, 192, 256])
    ms = p.parse_args().ms
    g = torch.Generator(device="cuda").manual_seed(0)
    ws = [(torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16) for _ in range(3)]
    print(f"{'M':>4} {'stock us':>9} {'hipBLASLt':>10} {'this us':>8} {'vs stock':>9} {'TB/s':>6} {'TFLOPs':>7} "
          f"{'ceiling us':>10} {'% ceiling':>9}")
    for m in ms:
        xs = [torch.randn(m, K, device="cuda").to(torch.bfloat16) for _ in range(6)]
        stock = replay_us(lambda i: tgemm.mm(xs[i % 6], ws[i % 3], None), 6)
        lt = replay_us(lambda i: torch.nn.functional.linear(xs[i % 6], ws[i % 3]), 6)
        mine = replay_us(lambda i: bf16_gemm_n20480_k7168(xs[i % 6], ws[i % 3]), 6)
        by = N * K * 2 + m * K * 2 + m * N * 2
        fl = 2 * m * N * K
        ceil = max(by / (HBM_TBS * 1e6), fl / (PEAK_TFLOPS * 1e6))
        print(f"{m:>4} {stock:>9.1f} {lt:>10.1f} {mine:>8.1f} {stock / mine:>8.2f}x {by / mine / 1e6:>6.2f} "
              f"{fl / mine / 1e6:>7.0f} {ceil:>10.1f} {100 * ceil / mine:>8.0f}%")


if __name__ == "__main__":
    main()
