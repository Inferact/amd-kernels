# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the ``markov_gumbel_fused`` entry adds over the package's call: graph replay us (8 chains per graph, each on
its own base logits; entry and direct graphs alternate) and host us per eager call (the enqueue loop of 20 calls, no
synchronization inside), entry vs direct, min of 20 rounds; and the host us of
``markov_gumbel_fused_blocker`` (the caller runs it once per call).
Run: python3 benchmarks/sampling/markov_gumbel_common/benchmark_routing.py
"""

import importlib
import time

import torch

from amd_kernels.sampling.markov_gumbel_common import (
    markov_gumbel_fused,
    markov_gumbel_fused_blocker,
    markov_scratch_dwords,
)
from amd_kernels.sampling.markov_gumbel_common.host import _ROUTES

V, RANK = 163840, 512
NIN, CALLS, ROUNDS = 8, 20, 20
CASES = ((8, 4), (20, 3), (48, 3))


def make(n, k, w1, w2, gen):
    slots = 64
    return {
        "w1": w1, "w2": w2, "input_ids": torch.randint(0, V, (n + 5,), generator=gen, dtype=torch.int32).cuda(),
        "anchor_idx": torch.arange(n).cuda(), "draft_tokens": torch.zeros(n, k, dtype=torch.int64, device="cuda"),
        "idx_mapping": torch.arange(n).view(n, 1).repeat(1, k).cuda(),
        "sample_pos": torch.randint(1, 200000, (n, k), generator=gen).cuda(),
        "temperature": torch.ones(slots, device="cuda"),
        "seeds": torch.randint(0, 2**62, (slots,), generator=gen).cuda(),
        "cache": torch.zeros(slots, k, V, dtype=torch.bfloat16, device="cuda"),
        "scratch": torch.zeros(markov_scratch_dwords(w1, n), dtype=torch.int32, device="cuda"), "row_mapping": None,
    }


def capture(fn, bases, kw):
    fn(bases[0], **kw)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for b in bases:
            fn(b, **kw)
    return g


def replay_us(graphs):
    best = [float("inf")] * len(graphs)
    for _ in range(ROUNDS):
        for i, g in enumerate(graphs):
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            e.record()
            e.synchronize()
            best[i] = min(best[i], a.elapsed_time(e) * 1000 / NIN)
    return best


def host_us(fn, *args, **kw):
    best = float("inf")
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(CALLS):
            fn(*args, **kw)
        best = min(best, (time.perf_counter() - t) * 1e6 / CALLS)
    torch.cuda.synchronize()
    return best


def main():
    pkg = importlib.import_module(_ROUTES.hosts[(V, RANK)])
    gen = torch.Generator().manual_seed(0)
    w1, w2 = ((torch.randn(V, RANK, generator=gen) * 0.05).bfloat16().cuda() for _ in range(2))
    print("| rows, steps | replay direct us | replay entry us | host direct us | host entry us | added us | reason us |")
    print("|---|---|---|---|---|---|---|")
    for n, k in CASES:
        kw = make(n, k, w1, w2, gen)
        bases = [(torch.randn(n, k, V, generator=gen) * 3).bfloat16().cuda() for _ in range(NIN)]
        assert markov_gumbel_fused_blocker(bases[0], **kw) is None
        rd, re = replay_us([capture(pkg.markov_gumbel_fused, bases, kw), capture(markov_gumbel_fused, bases, kw)])
        hd, he = host_us(pkg.markov_gumbel_fused, bases[0], **kw), host_us(markov_gumbel_fused, bases[0], **kw)
        hr = host_us(markov_gumbel_fused_blocker, bases[0], **kw)
        print(f"| {n}, {k} | {rd:.2f} | {re:.2f} | {hd:.2f} | {he:.2f} | {he - hd:+.2f} | {hr:.2f} |")


if __name__ == "__main__":
    main()
