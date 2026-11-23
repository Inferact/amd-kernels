# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared-expert down GEMM -> routed add -> mHC seam chains per row count, graph replay A B B A: the fused chains against
the separate one.

Chains of one FFN boundary, all from the gate_up epilogue's MXFP8 codes, the routed experts' bf16 output and the previous
seam's gates:
  base     shared down GEMM to bf16 (the faster of the decode GEMM on row-major codes and the staged GEMM), ``routed +
           shared`` (one elementwise launch) then the seam with the post-mix (this repository's seam kernel or AITER's
           ``mega_mhc``, the faster), MXFP8 output
  plain    down GEMM with the routed add and the post-mix epilogue, then the seam without post-mix
  gates    down GEMM with the add, the post-mix and the gate-partial epilogue, then the finisher (MXFP8 output)

100 distinct inputs per graph, the weight in 10 cold copies, 5 rounds of 5 replays, median us per chain.

  python3 benchmarks/gemm/mxfp8_postmix_add_hc4_n5120_k2304/benchmark_chain.py [--ms 16,192] [--out result.json]
"""

import argparse
import json
import statistics

import torch
from aiter.ops.mega_mhc import mega_mhc, mega_mhc_shuffle_fn

from amd_kernels.gemm import mxfp8_postmix_add_hc4_n5120_k2304 as pm
from amd_kernels.gemm.mxfp8_common import gemm as common
from amd_kernels.gemm.mxfp8_decode_prequant import Config as DecodeConfig
from amd_kernels.gemm.mxfp8_decode_prequant import gemm as decode_gemm
from amd_kernels.gemm.mxfp8_postmix_add_hc4_n5120_k2304.reference import quantize_mxfp8
from amd_kernels.norm.mhc_gate_finish_rmsnorm_h5120 import mhc_gate_finish
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import mhc_post_pre_rmsnorm
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import pack_gate_weight as pack_seam

N = H = 5120
K = 2304
COUNT, COPIES = 100, 10
ROWS = (16, 32, 64, 128, 160, 192, 256, 320, 384, 512, 640, 768)
# fastest decode-GEMM (rt, ni, kw, ks, weight cache modifier) per row count on row-major codes (sweep of rt 1-8, ni 2-8,
# (kw, ks) in 1, 3, 9, cache modifier 0 / 1 / 2)
DECODE = {16: (1, 2, 1, 1, 2), 32: (2, 2, 1, 1, 2), 64: (1, 5, 3, 1, 0), 128: (3, 4, 1, 1, 2), 160: (3, 5, 3, 1, 1),
          192: (4, 4, 1, 1, 2), 256: (4, 5, 3, 1, 0), 320: (4, 8, 3, 1, 0), 384: (4, 8, 3, 1, 0), 512: (4, 5, 1, 1, 0),
          640: (6, 5, 1, 1, 2), 768: (6, 5, 1, 1, 1)}  # fmt: skip
SEAM = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.5, norm_eps=1e-6)


def capture(fn, stream):
    """Warm up and capture ``COUNT`` calls on ``stream`` (Mega-MHC's scratch is per capture stream)."""
    with torch.cuda.stream(stream):
        fn(0)
        fn(COUNT - 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for j in range(COUNT):
            fn(j)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def replay_us(graph, reps=5):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * COUNT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", default=",".join(map(str, ROWS)))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()
    dev = "cuda"
    stream = torch.cuda.Stream()
    gen = torch.Generator(device=dev).manual_seed(0)
    codes, exps = quantize_mxfp8((torch.randn(N, K, generator=gen, device=dev) * 0.05).bfloat16())
    w, ws = pm.preshuffle_weight(codes, exps)
    weights = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(COPIES - 1)]
    fn = torch.randn(24, 4 * H, generator=gen, device=dev) * 0.02
    fn_gemm, fn_seam, fn_aiter = pm.pack_gate_weight(fn), pack_seam(fn), mega_mhc_shuffle_fn(fn)
    sc, base = torch.tensor([0.5, 0.7, 0.9], device=dev), torch.randn(24, generator=gen, device=dev) * 0.3
    nw = torch.randn(H, generator=gen, device=dev).bfloat16()
    result = {}
    for m in map(int, args.ms.split(",")):
        xs = [quantize_mxfp8(torch.randn(m, K, generator=gen, device=dev).bfloat16()) for _ in range(COUNT)]
        res = [torch.randn(m, 4, H, generator=gen, device=dev).bfloat16() for _ in range(COUNT)]
        routed = [torch.randn(m, H, generator=gen, device=dev).bfloat16() for _ in range(COUNT)]
        post = torch.rand(m, 4, 1, generator=gen, device=dev)
        comb = torch.softmax(torch.randn(m, 4, 4, generator=gen, device=dev), -1)
        pre = torch.rand(m, 4, generator=gen, device=dev)
        cfg = DecodeConfig(*DECODE[m][:4], 0, DECODE[m][4])

        def y_decode(j):
            wj, wsj = weights[j % COPIES]
            return decode_gemm(xs[j][0], xs[j][1], wj, wsj, cfg)

        def y_staged(j):
            wj, wsj = weights[j % COPIES]
            return common.mxfp8_gemm(xs[j][0], xs[j][1], wj, wsj, torch.bfloat16)

        def seam_ours(j, y):
            y = y + routed[j]
            return mhc_post_pre_rmsnorm(res[j], fn_seam, sc, base, nw, pre_mix=pre, sublayer_out=y, post_mix=post,
                                        comb_mix=comb, quant=True, **SEAM)

        def seam_aiter(j, y):
            y = y + routed[j]
            return mega_mhc(res[j], fn_aiter, sc, base, 1e-6, 1e-6, 1e-6, 1.5, 20, pre, y, post, comb, nw, 1e-6,
                            residual_out=torch.empty_like(res[j]), w_preshuffle_bf16=True, out_dtype="mxfp8")

        def plain(j):
            wj, wsj = weights[j % COPIES]
            r = pm.postmix_gemm(xs[j][0], xs[j][1], wj, wsj, res[j], routed[j], post, comb)
            return mhc_post_pre_rmsnorm(r, fn_seam, sc, base, nw, pre_mix=pre, quant=True, **SEAM)

        def gates(j):
            wj, wsj = weights[j % COPIES]
            r, x1, gp, _ = pm.postmix_gates_gemm(xs[j][0], xs[j][1], wj, wsj, res[j], routed[j], post, comb, pre, fn_gemm)
            return mhc_gate_finish(gp, x1, sc, base, nw, quant=True, **SEAM)

        chains = {
            "base decode+seam": lambda j: seam_ours(j, y_decode(j)),
            "base decode+aiter": lambda j: seam_aiter(j, y_decode(j)),
            "base staged+seam": lambda j: seam_ours(j, y_staged(j)),
            "base staged+aiter": lambda j: seam_aiter(j, y_staged(j)),
            "plain": plain,
            "gates": gates,
        }
        graphs = {name: capture(fn_, stream) for name, fn_ in chains.items()}
        times = {name: [] for name in graphs}
        order = list(graphs)
        for name in (order + order[::-1]) * args.rounds:
            times[name].append(replay_us(graphs[name]))
        med = {name: statistics.median(v) for name, v in times.items()}
        best = min(v for k, v in med.items() if k.startswith("base"))
        result[m] = dict(med, base_best=best, plain_vs=med["plain"] / best, gates_vs=med["gates"] / best)
        print(f"{m:4d} " + " ".join(f"{k}={v:.2f}" for k, v in med.items()) + f" | base {best:.2f}  plain/base "
              f"{med['plain'] / best:.3f}  gates/base {med['gates'] / best:.3f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
