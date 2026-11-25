# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the ``top_p_mask`` entry adds over the vocabulary package's call: graph replay us (40 masks per graph, each on
its own rows, restored before every replay; entry and direct graphs alternate) and host us per eager call (the
enqueue loop of 100 calls, no synchronization inside), entry vs direct, min of 20 rounds; and the host us of
``top_p_mask_blocker`` (the caller runs it once per call).
Run: python3 benchmarks/sampling/top_p_mask_common/benchmark_routing.py
"""

import importlib
import time

import torch

from amd_kernels.sampling.top_p_mask_common import top_p_mask, top_p_mask_blocker
from amd_kernels.sampling.top_p_mask_common.host import _ROUTES

NIN, CALLS, ROUNDS = 40, 100, 20
CASES = ((163840, 8), (163840, 64), (200064, 100))


def capture(fn, bufs, p, buffer):
    for b in bufs[:2]:
        fn(b, p, buffer)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for b in bufs:
            fn(b, p, buffer)
    return g


def replay_us(graphs, bufs, src):
    best = [float("inf")] * len(graphs)
    for _ in range(ROUNDS):
        for i, g in enumerate(graphs):
            for b, s in zip(bufs, src):
                b.copy_(s)
            a, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            e.record()
            e.synchronize()
            best[i] = min(best[i], a.elapsed_time(e) * 1000 / len(bufs))
    return best


def host_us(fn, *args):
    best = float("inf")
    for _ in range(ROUNDS):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(CALLS):
            fn(*args)
        best = min(best, (time.perf_counter() - t) * 1e6 / CALLS)
    torch.cuda.synchronize()
    return best


def main():
    print("| vocab, rows | replay direct us | replay entry us | host direct us | host entry us | added us | reason us |")
    print("|---|---|---|---|---|---|---|")
    for vocab, n in CASES:
        pkg = importlib.import_module(_ROUTES.hosts[(vocab,)])
        buffer = torch.empty(pkg.rows_grid(n, 0), vocab, device="cuda") if n > 64 else None

        def direct(x, p, buf, pkg=pkg):
            return pkg.top_p_mask(x, p) if buf is None else pkg.top_p_mask(x, p, buf)

        gen = torch.Generator().manual_seed(n)
        src = [(torch.randn(n, vocab, generator=gen) * 3).bfloat16().cuda() for _ in range(NIN)]
        bufs = [s.float() for s in src]
        p = torch.full((n,), 0.9, device="cuda")
        assert top_p_mask_blocker(bufs[0], p, buffer) is None
        rd, re = replay_us([capture(direct, bufs, p, buffer), capture(top_p_mask, bufs, p, buffer)], bufs, src)
        hd, he = host_us(direct, bufs[0], p, buffer), host_us(top_p_mask, bufs[0], p, buffer)
        hr = host_us(top_p_mask_blocker, bufs[0], p, buffer)
        print(f"| {vocab}, {n} | {rd:.2f} | {re:.2f} | {hd:.2f} | {he:.2f} | {he - hd:+.2f} | {hr:.2f} |")


if __name__ == "__main__":
    main()
