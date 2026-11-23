# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""mHC seam graph latency vs AITER's HIP Mega-MHC; 100 distinct inputs per graph, rounds A B B A."""
import statistics

import torch
from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn
from aiter.ops.mhc import mhc_post
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import mhc_post_pre_rmsnorm, pack_gate_weight

H, N, ROUNDS = 5120, 100, 5
ROWS = (16, 32, 64, 128, 256, 384, 512, 768)


def graph_us(fn, stream):
    with torch.cuda.stream(stream):
        fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
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
    fn = torch.randn(24, 4 * H, device=dev) * 0.02
    fnp, fnb = pack_gate_weight(fn), mega_mhc_shuffle_fn(fn)
    sc, base = torch.tensor([0.5, 0.7, 0.9], device=dev), torch.randn(24, device=dev) * 0.3
    w = torch.randn(H, device=dev).bfloat16()
    stream = torch.cuda.Stream()
    kw = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.5, norm_eps=1e-6)
    print("rows  mode   baseline_us   ours_us   (median [min-max] over rounds)")
    for m in ROWS:
        R = [torch.randn(m, 4, H, device=dev).bfloat16() for _ in range(N)]
        y = [torch.randn(m, H, device=dev).bfloat16() for _ in range(N)]
        pm, pre = torch.rand(m, 4, device=dev), torch.rand(m, 4, device=dev)
        cm = torch.softmax(torch.randn(m, 4, 4, device=dev), -1)
        for mode in ("bf16", "mxfp8", "mxfp8+aux"):
            aux = mode.endswith("aux")

            def ours(i):
                mhc_post_pre_rmsnorm(R[i], fnp, sc, base, w, pre_mix=pre, sublayer_out=y[i], post_mix=pm, comb_mix=cm,
                                     quant=mode != "bf16", aux_mean=aux, **kw)

            def theirs(i):
                out = mega_mhc(R[i], fnb, sc, base, 1e-6, 1e-6, 1e-6, 1.5, 20, pre, y[i], pm, cm, w, 1e-6,
                               residual_out=torch.empty_like(R[i]), w_preshuffle_bf16=True,
                               out_dtype="bf16" if mode == "bf16" else "mxfp8")
                if aux:
                    r = torch.empty_like(R[i])
                    mhc_post(r, y[i], R[i], pm, cm)
                    r.mean(dim=1)
                return out

            gs = {"ours": graph_us(ours, stream), "base": graph_us(theirs, stream)}
            ts = {"ours": [], "base": []}
            for k in ("ours", "base", "base", "ours") * ROUNDS:
                ts[k].append(replay_us(gs[k]))
            f = lambda v: f"{statistics.median(v):7.2f} [{min(v):.2f}-{max(v):.2f}]"  # noqa: E731
            print(f"{m:5d} {mode:10s} {f(ts['base'])} {f(ts['ours'])}", flush=True)


if __name__ == "__main__":
    main()
