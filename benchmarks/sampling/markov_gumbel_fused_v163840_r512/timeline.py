# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Per-step timeline of the fused chain from the kernel's own timestamps (``s_memrealtime``, 100 MHz, wave 0 of every
workgroup, written to the scratch with ``AMD_KERNELS_MARKOV_TS=1``), under the benchmark's conditions (graph replay,
rotated ``markov_w2`` copies and inputs). Per step: the tile loop's start, its end (min / median / max over the 256
workgroups: the spread), the last aggregation, and the tokens at workgroup 0.
Run: python3 benchmarks/sampling/markov_gumbel_fused_v163840_r512/timeline.py 8 4   (requests, steps)
"""

import os
import sys

os.environ["AMD_KERNELS_MARKOV_TS"] = "1"

import numpy as np
import torch

from amd_kernels.sampling.markov_gumbel_fused_v163840_r512 import (
    RANK,
    V,
    kernel,
    markov_gumbel_fused,
    scratch_dwords,
)


def main():
    n, k = int(sys.argv[1]), int(sys.argv[2])
    dev = torch.device("cuda", 0)
    gen = torch.Generator().manual_seed(1)
    w1 = (torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().to(dev)
    w2s = [(torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().to(dev) for _ in range(3)]
    sets = [
        {
            "base": (torch.randn(n, k, V, generator=gen) * 3).bfloat16().to(dev),
            "ids": torch.randint(0, V, (n + 4,), generator=gen, dtype=torch.int32).to(dev),
            "pos": torch.randint(1000, 200000, (n, k), generator=gen).to(dev),
            "toks": torch.zeros(n, k, dtype=torch.int64, device=dev),
            "cache": torch.zeros(n + 3, k, V, dtype=torch.bfloat16, device=dev),
        }
        for _ in range(8)
    ]
    anchor = torch.arange(n, device=dev) + 1
    idx = torch.arange(n, device=dev).view(n, 1).repeat(1, k).contiguous()
    temp = torch.ones(n + 3, dtype=torch.float32, device=dev)
    seeds = torch.arange(n + 3, dtype=torch.int64, device=dev) + 7
    scratch = torch.zeros(scratch_dwords(n) + 256 * k * 24, dtype=torch.int32, device=dev)

    def call(i):
        d = sets[i % 8]
        markov_gumbel_fused(d["base"], w1, w2s[i % 3], d["ids"], anchor, d["toks"], idx, d["pos"], temp, seeds,
                            d["cache"], scratch)

    call(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(8):
            call(i)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    raw = scratch[kernel.TS_OFF:kernel.TS_OFF + 256 * k * 24].cpu().numpy().astype(np.int64).reshape(256, k, 24)
    rel = (((raw & 0xFFFFFFFF) - (raw[:, 0, 19] & 0xFFFFFFFF).min()) & 0xFFFFFFFF).astype(np.float64) * 0.01  # us
    print(f"{n} requests, {k} steps, last of 8 chained calls; us from the first workgroup's start")
    print("step | loop start | loop end min / med / max | spread | last aggregation | tokens | tokens - loop end max")
    for s in range(k):
        start, end, agg, tok = np.median(rel[:, s, 1]), rel[:, s, 2], rel[:, s, 4].max(), rel[:, s, 5].max()
        tok = rel[0, s, 5] if s == k - 1 else tok
        print(f"{s:4d} | {start:10.1f} | {end.min():7.1f} {np.median(end):7.1f} {end.max():7.1f} | {end.max() - end.min():6.1f} | "
              f"{agg:16.1f} | {tok:6.1f} | {tok - end.max():6.1f}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
