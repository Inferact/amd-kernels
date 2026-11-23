# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Step-end mHC collapse + RMSNorm graph latency vs the served chain; 100 distinct inputs per graph, rounds A B B A."""
import statistics

import aiter
import torch
from aiter.ops.mhc import mhc_post
from amd_kernels.norm.mhc_post_collapse_rmsnorm_h5120 import mhc_post_collapse_rmsnorm
from vllm.model_executor.kernels.mhc.triton import hc_collapse_triton, mhc_post_collapse_rms_norm_triton

H, N, ROUNDS = 5120, 100, 5
ROWS = (16, 32, 64, 128, 256, 384, 512, 768)


def graph(fn):
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(N):
            fn(i)
    return g


def replay_us(g):
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (5 * N)


def main():
    torch.manual_seed(0)
    dev = "cuda"
    w = torch.randn(H, device=dev).bfloat16()
    f = lambda v: f"{statistics.median(v):7.2f} [{min(v):.2f}-{max(v):.2f}]"  # noqa: E731
    print("rows  case        served_chain   fused_triton   ours   (us, median [min-max])")
    for m in ROWS:
        R = [torch.randn(m, 4, H, device=dev).bfloat16() for _ in range(N)]
        y = [torch.randn(m, H, device=dev).bfloat16() for _ in range(N)]
        pm, pre = torch.rand(m, 4, 1, device=dev), torch.rand(m, 4, device=dev)
        cm = torch.softmax(torch.randn(m, 4, 4, device=dev), -1)

        def chain(i):
            r = torch.empty_like(R[i])
            mhc_post(r, y[i], R[i], pm, cm)
            return aiter.rms_norm(hc_collapse_triton(r, pre), w, 1e-6)

        def fused(i):
            return mhc_post_collapse_rms_norm_triton(R[i], y[i], pm, cm, pre, w, 1e-6)

        def ours(i):
            return mhc_post_collapse_rmsnorm(R[i], pre, w, eps=1e-6, sublayer_out=y[i], post_mix=pm, comb_mix=cm)

        def chain_aux(i):
            r = torch.empty_like(R[i])
            mhc_post(r, y[i], R[i], pm, cm)
            r.mean(dim=1)
            return aiter.rms_norm(hc_collapse_triton(r, pre), w, 1e-6)

        def ours_aux(i):
            return mhc_post_collapse_rmsnorm(R[i], pre, w, eps=1e-6, sublayer_out=y[i], post_mix=pm, comb_mix=cm,
                                             aux_mean=True)

        for case, a, b, c in (("post+tail", chain, fused, ours), ("post+tail+aux", chain_aux, None, ours_aux)):
            gs = {"a": graph(a), "c": graph(c)}
            if b:
                gs["b"] = graph(b)
            ts = {k: [] for k in gs}
            for k in ("a", "b", "c", "c", "b", "a") * ROUNDS:
                if k in gs:
                    ts[k].append(replay_us(gs[k]))
            print(f"{m:5d} {case:13s} {f(ts['a'])} {f(ts['b']) if b else '       -       '} {f(ts['c'])}", flush=True)


if __name__ == "__main__":
    main()
