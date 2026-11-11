# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``top_p_mask`` per row count (the sampled rows of one TP4 rank: requests x (1 + draft
tokens) / 4): 100 distinct inputs per graph, each a bf16 -> fp32 copy of LM-like logits and the mask (the served
step: the mask is in place, so every replay restarts from the bf16 rows); the copy alone is timed as well and
subtracted. Median of 5 rounds x 5 replays, us per call. ``--rows <file.pt>``: a saved [R, 200064] tensor of real
logits rows; each input then takes random rows of it at a temperature in 0.85-1.15 (the README's table); without it
the rows are synthesized.
Run: python3 benchmarks/sampling/top_p_mask_f32_v200064/benchmark_top_p_mask.py [--rows <file.pt>]
"""

import argparse
import statistics

import torch
from amd_kernels.sampling.top_p_mask_f32_v200064 import V, splits_for, top_p_mask, warmup_top_p_mask

NIN = 100


def timed(g):
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * NIN))
    return statistics.median(ts)


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", help="a saved [R, 200064] tensor of real logits rows")
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    warmup_top_p_mask(dev)
    gen = torch.Generator().manual_seed(1)
    real = None
    if args.rows:
        real = torch.load(args.rows, weights_only=True).float()
        assert real.dim() == 2 and real.shape[1] == V, real.shape
    print("rows | slices | copy us | copy + mask us | mask us")
    for n in (1, 4, 8, 16, 32, 64):
        src = []
        for _ in range(NIN):
            if real is not None:
                t = torch.rand(n, 1, generator=gen) * 0.3 + 0.85
                x = real[torch.randint(0, real.shape[0], (n,), generator=gen)] / t
            else:
                x = torch.randn(n, V, generator=gen) * 2.0
                x.scatter_(1, torch.randint(0, V, (n, 64), generator=gen), torch.rand(n, 64, generator=gen) * 12 + 6)
            src.append(x.to(torch.bfloat16).to(dev))
        bufs = [torch.empty(n, V, device=dev) for _ in range(NIN)]
        p = torch.full((n,), 0.95, device=dev)

        def copy():
            for b, s in zip(bufs, src):
                b.copy_(s)

        def step():
            for b, s in zip(bufs, src):
                b.copy_(s)
                top_p_mask(b, p)

        gc, gs = graph(copy), graph(step)
        tc, ts = [], []
        for _ in range(2):
            tc.append(timed(gc))
            ts.append(timed(gs))
        c, t = statistics.median(tc), statistics.median(ts)
        print(f"{n} | {splits_for(n)} | {c:.1f} | {t:.1f} | {t - c:.1f}", flush=True)
        del gc, gs, bufs, src
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
