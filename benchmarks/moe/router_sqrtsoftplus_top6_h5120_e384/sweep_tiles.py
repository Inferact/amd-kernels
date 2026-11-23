# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Random sweep of the GEMM tiles (``host._TILES``): per row count, graph-replay time of the whole entry for ``--n``
random (a, b, ksp, kw, chunk, xm, xn) with 64-2048 workgroups and at most 64 KB of LDS; prints the six fastest.
Run: python3 benchmarks/moe/router_sqrtsoftplus_top6_h5120_e384/sweep_tiles.py --rows 16 64 192 --n 60
"""

import argparse
import random
import statistics

import torch
from amd_kernels import warm_up
from amd_kernels.moe.router_sqrtsoftplus_top6_h5120_e384 import host

NIN = 100


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


def timed(g):
    ts = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * NIN))
    return statistics.median(ts)


def space(m):
    out = []
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4, 6):
            for xm, xn in ((1, 8), (2, 4), (4, 2), (8, 1)):
                ng = 24 // b
                if ng % xn:
                    continue
                for ksp in (1, 2, 4, 5, 8, 10, 16, 20):
                    for kw in (1, 2, 4, 8):
                        if 160 % (ksp * kw):
                            continue
                        spw = 160 // ksp // kw
                        chunk = max(c for c in (1, 2, 4, 5, 8, 10) if spw % c == 0 and c <= max(2, 40 // (a + b)))
                        mgs = -(-(-(-m // (16 * a))) // xm)
                        wgs = 8 * mgs * ksp * (ng // xn)
                        if 64 <= wgs <= 2048 and kw * 256 * a * b * 4 <= 65536 and (a + b) * chunk * 4 <= 120:
                            out.append((a, b, ksp, kw, chunk, xm, xn))
    return out


def main():
    warm_up.set_auto(False)  # the sweep calls variants the warm-up does not know
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[16, 64, 192, 384, 768])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    w = (torch.randn(384, 5120, device="cuda") * 0.02).bfloat16()
    bias = torch.randn(384, device="cuda") * 0.1
    for m in args.rows:
        xs = [torch.randn(m, 5120, device="cuda").bfloat16() for _ in range(NIN)]
        outs = [(torch.empty(m, 6, device="cuda"), torch.empty(m, 6, device="cuda", dtype=torch.int32)) for _ in range(NIN)]
        cands = space(m)
        rng.shuffle(cands)
        res = []
        for tile in cands[: args.n]:
            host._tile = lambda rows, tile=tile: tile

            def fn():
                for x, (tw, ti) in zip(xs, outs):
                    host.router_topk(x, w, bias, 1.5, topk_weights=tw, topk_ids=ti)

            g = graph(fn)
            res.append((min(timed(g), timed(g)), tile))
            host._gemm_exe.cache_clear()
            host._topk_exe.cache_clear()
        res.sort()
        print(m, [(round(t, 2), tile) for t, tile in res[:6]], flush=True)


if __name__ == "__main__":
    main()
