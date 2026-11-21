# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``markov_chain`` against vLLM's per-step loop (Markov embedding, bias GEMM, ``gumbel_sample``
with the fused bias) per request count: ~100 distinct base-logit sets per graph (bigger than the LLC), 5 alternated
rounds of 5 replays in the order baseline, entry, entry, baseline. Needs vLLM for the baseline.
Run: python3 benchmarks/sampling/markov_chain_gumbel_bf16_v129280_r256_s5/benchmark_markov_chain.py [n ...]
"""

import sys

import torch
import torch.nn.functional as Fn
from amd_kernels.sampling.markov_chain_gumbel_bf16_v129280_r256_s5 import NSPEC, R, V, markov_chain, warmup_markov_chain

SLOTS = 512
MAX_SETS = 100
BUDGET = 40 << 30  # bytes of base logits


def baseline(base, w1, w2, ids, anchor, idx, pos, temp, seeds, cache, tokens):
    from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

    n = anchor.shape[0]
    cols = torch.arange(NSPEC, dtype=torch.int32, device=base.device)
    base3 = base.view(n, NSPEC, V)
    prev = ids[anchor]
    for s in range(NSPEC):
        bias = Fn.linear(Fn.embedding(prev, w1), w2)
        out = gumbel_sample(base3[:, s], idx[:, s], temp, seeds, pos[:, s] - 1, apply_temperature=True,
                            is_drafting=True, logits_cache=cache, logits_cache_col=cols[s], bias=bias)
        tokens[:n, s] = out
        prev = out


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


def timed(g, calls, replays=5):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(replays):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (replays * calls)


def main(rows):
    dev = torch.device("cuda", 0)
    warmup_markov_chain(dev)
    gen = torch.Generator().manual_seed(1)
    w1 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).to(dev)
    w2 = (torch.randn(V, R, generator=gen) * 0.3).to(torch.bfloat16).to(dev)
    ids = torch.randint(0, V, (1 << 16,), generator=gen).to(torch.int32).to(dev)
    temp = torch.ones(SLOTS, dtype=torch.float32, device=dev)
    seeds = torch.randint(-(2**62), 2**62, (SLOTS,), generator=gen).to(dev)
    cache = torch.zeros(SLOTS, NSPEC, V, dtype=torch.bfloat16, device=dev)
    print("rows | baseline us | entry us | speedup  (median [min-max] of 5 rounds, us per 5-step chain)")
    for n in rows:
        count = max(4, min(MAX_SETS, BUDGET // (n * NSPEC * V * 2)))
        sets = []
        for _ in range(count):
            sets.append(dict(
                base=(torch.randn(n * NSPEC, V, generator=gen) * 3).to(torch.bfloat16).to(dev),
                anchor=torch.randint(0, 1 << 16, (n,), generator=gen).to(dev),
                idx=torch.randperm(SLOTS, generator=gen)[:n].view(n, 1).expand(n, NSPEC).contiguous().to(dev),
                pos=(131072 + torch.randint(0, 4096, (n, 1), generator=gen) + torch.arange(NSPEC)).to(dev),
            ))
        tokens = torch.zeros(n, NSPEC, dtype=torch.int64, device=dev)

        def make(fn):
            def calls():
                for d in sets:
                    fn(d["base"], w1, w2, ids, d["anchor"], d["idx"], d["pos"], temp, seeds, cache, tokens)
            return calls

        g_base, g_new = graph(make(baseline)), graph(make(markov_chain))
        tb, tn = [], []
        for _ in range(5):
            tb.append(timed(g_base, count))
            tn.append(timed(g_new, count))
            tn.append(timed(g_new, count))
            tb.append(timed(g_base, count))
        tb.sort()
        tn.sort()
        mb, mn = tb[len(tb) // 2], tn[len(tn) // 2]
        print(f"{n:4d} | {mb:.1f} [{tb[0]:.1f}-{tb[-1]:.1f}] | {mn:.1f} [{tn[0]:.1f}-{tn[-1]:.1f}] | {mb / mn:.2f}x",
              flush=True)
        del sets, g_base, g_new


if __name__ == "__main__":
    with torch.inference_mode():
        main([int(a) for a in sys.argv[1:]] or [8, 16, 32, 48, 64, 96, 128, 160])
