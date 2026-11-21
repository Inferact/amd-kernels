# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the fused Markov chain per served draft config (requests, steps) against vLLM's stock loop
(embedding gather, ``rocm_unquantized_gemm`` over ``markov_w2``, base + bias, ``gumbel_sample`` with the draft logits
cache, copy into ``draft_tokens``) and against two measured ceilings (``skeleton.py``): A = k launches of the pure
``markov_w2`` stream, B = one persistent launch of k stream passes without step synchronization, each plus the base-logit
read and cache write traffic at the measured 6.1 TB/s. 3 rotated ``markov_w2`` copies and 8 input sets per graph
(HBM-cold), 10 replays, alternated rounds, median and min us per chain.
Run: python3 benchmarks/sampling/markov_gumbel_fused_v163840_r512/benchmark_markov_gumbel_fused.py [--temp 1.0] [--rounds 9]
"""

import argparse
import statistics

import torch
from skeleton import run_stream

from amd_kernels.sampling.markov_gumbel_fused_v163840_r512 import (
    RANK,
    V,
    markov_gumbel_fused,
    scratch_dwords,
)

TRAFFIC_TBPS = 6.1  # measured: base-logit read + cache write mix
CONFIGS = ((8, 4), (13, 4), (20, 3), (48, 3))  # (requests, steps): c16, c24, c32, c48


def make_sets(n, k, gen, dev):
    return [
        {
            "base": (torch.randn(n, k, V, generator=gen) * 3).bfloat16().to(dev),
            "ids": torch.randint(0, V, (n + 4,), generator=gen, dtype=torch.int32).to(dev),
            "pos": torch.randint(1000, 200000, (n, k), generator=gen).to(dev),
            "toks": torch.zeros(n, k, dtype=torch.int64, device=dev),
            "cache": torch.zeros(n + 3, k, V, dtype=torch.bfloat16, device=dev),
        }
        for _ in range(8)
    ]


def replay_graph(fn, count):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(count):
            fn(i)
    g.replay()
    return g


def bench_config(n, k, temp, w1, w2s, gen, dev, rounds):
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl as gemm
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    sets = make_sets(n, k, gen, dev)
    anchor = torch.arange(n, device=dev) + 1
    idx = torch.arange(n, device=dev).view(n, 1).repeat(1, k).contiguous()
    tmp = torch.full((n + 3,), temp, dtype=torch.float32, device=dev)
    seeds = torch.arange(n + 3, dtype=torch.int64, device=dev) + 7
    scratch = torch.zeros(scratch_dwords(n), dtype=torch.int32, device=dev)
    cols = torch.arange(k, dtype=torch.int32, device=dev)
    sink = torch.zeros(1 << 20, dtype=torch.int32, device=dev)

    def fused(i):
        d = sets[i % len(sets)]
        markov_gumbel_fused(d["base"], w1, w2s[i % 3], d["ids"], anchor, d["toks"], idx, d["pos"], tmp, seeds,
                            d["cache"], scratch)

    def stock(i):
        d = sets[i % len(sets)]
        prev = d["ids"][anchor].long()
        for s in range(k):
            bias = gemm(torch.nn.functional.embedding(prev, w1), w2s[i % 3])
            prev = gumbel_sample(d["base"][:, s] + bias, idx[:, s], tmp, seeds, d["pos"][:, s] - 1,
                                 apply_temperature=True, is_drafting=True, logits_cache=d["cache"],
                                 logits_cache_col=cols[s], use_fp64=False)
            d["toks"][:, s] = prev

    def ceil_a(i):
        for _ in range(k):
            run_stream(1, w2s[i % 3], sink, torch.cuda.current_stream())

    def ceil_b(i):
        run_stream(k, w2s[i % 3], sink, torch.cuda.current_stream())

    fns = (stock, fused, ceil_a, ceil_b)
    graphs = [replay_graph(fn, len(sets)) for fn in fns]
    torch.cuda.synchronize()
    ts = [[] for _ in fns]
    for r in range(rounds):
        for j in (range(len(fns)) if r % 2 == 0 else reversed(range(len(fns)))):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(10):
                graphs[j].replay()
            b.record()
            b.synchronize()
            ts[j].append(a.elapsed_time(b) * 1e3 / (10 * len(sets)))
    t_stock, t_fused, t_a, t_b = (statistics.median(x) for x in ts)
    m_fused = min(ts[1])
    traffic = k * 2 * n * V * 2 / (TRAFFIC_TBPS * 1e6)  # us
    ca, cb = t_a + traffic, t_b + traffic
    hbm = k * (V * RANK * 2 + 2 * n * V * 2)
    print(f"{n:4d} {k} | {t_stock:7.1f} | {t_fused:7.1f} ({m_fused:6.1f}) | {t_stock / t_fused:5.2f}x | {ca:6.1f} | "
          f"{100 * ca / t_fused:4.0f}% | {cb:6.1f} | {100 * cb / t_fused:4.0f}% | {hbm / t_fused / 1e6:5.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--rounds", type=int, default=9)
    args = p.parse_args()
    temp = args.temp
    dev = torch.device("cuda", 0)
    gen = torch.Generator().manual_seed(1)
    w1 = (torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().to(dev)
    w2s = [(torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().to(dev) for _ in range(3)]
    print(f"temperature {temp}")
    print("reqs k | stock us | fused us (min) | speedup | ceiling A us | % of A | ceiling B us | % of B | HBM TB/s")
    for n, k in CONFIGS:
        bench_config(n, k, temp, w1, w2s, gen, dev, args.rounds)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
