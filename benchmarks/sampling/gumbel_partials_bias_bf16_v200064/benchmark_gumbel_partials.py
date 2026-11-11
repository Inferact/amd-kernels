# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``gumbel_partials_bias`` per owned-row count (the Markov rows of 10-256 requests on one TP4
rank) and tiling: 20 distinct inputs per graph (HBM-cold, as each Markov step reads new rows), 10 replays, 5 rounds.
Run: python3 benchmarks/sampling/gumbel_partials_bias_bf16_v200064/benchmark_gumbel_partials.py
"""

import torch
from amd_kernels.sampling.gumbel_partials_bias_bf16_v200064 import (
    NB,
    V,
    blocks_per_wg,
    gumbel_partials_bias,
    warmup_gumbel_partials_bias,
)


def main():
    dev = torch.device("cuda", 0)
    warmup_gumbel_partials_bias(dev)
    gen = torch.Generator().manual_seed(1)
    print("rows | tiles 1 us | tiles 2 us | entry picks (us: median [min-max] of 5 rounds)")
    for n in (3, 4, 8, 16, 32, 64):
        sets = []
        for _ in range(20):
            sets.append(dict(
                base=(torch.randn(n, V, generator=gen) * 3).to(torch.bfloat16).to(dev),
                bias=(torch.randn(n, V, generator=gen) * 0.5).to(torch.bfloat16).to(dev),
                idx=torch.randperm(256, generator=gen)[:n].to(torch.int32).to(dev),
                pos=(131072 + torch.randint(0, 4096, (n,), generator=gen)).to(dev),
            ))
        temp = torch.ones(256, dtype=torch.float32, device=dev)
        seeds = torch.randint(-(2**62), 2**62, (256,), generator=gen).to(dev)
        rows = torch.arange(n, device=dev)
        cache = torch.zeros(256, 7, V, dtype=torch.bfloat16, device=dev)
        lm = torch.empty(n, NB, device=dev)
        la = torch.empty(n, NB, dtype=torch.int64, device=dev)
        res = {}
        for tiles in (1, 2):
            def calls(tiles=tiles):
                for k, d in enumerate(sets):
                    gumbel_partials_bias(d["base"], d["idx"], temp, seeds, d["pos"], cache, k % 7, d["bias"], rows,
                                         lm, la, tiles)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                calls()
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                calls()
            g.replay()
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(10):
                    g.replay()
                b.record()
                b.synchronize()
                ts.append(a.elapsed_time(b) * 1e3 / (10 * len(sets)))
            ts.sort()
            res[tiles] = f"{ts[2]:.2f} [{ts[0]:.2f}-{ts[-1]:.2f}]"
        print(f"{n:4d} | {res[1]} | {res[2]} | {blocks_per_wg(n)}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
