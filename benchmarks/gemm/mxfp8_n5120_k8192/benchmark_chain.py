# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the wo_a + wo_b pair per row count, A B B A: wo_a's MXFP8 epilogue + AITER's FlyDSL
``gemm_a8w8_blockscale_bpreshuffle`` on its codes (the served pair) against wo_a's bf16 output + ``mxfp8_linear``.

Random weights in cold copies, 100 distinct inputs per graph, 9 rounds of 5 replays, median us per pair.

  python3 benchmarks/gemm/mxfp8_n5120_k8192/benchmark_chain.py [--ms 769,848,1024]
"""

import argparse
import statistics

import torch
from aiter import gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

from amd_kernels.gemm import mxfp8_grouped_g8_n1024_k4096 as grouped
from amd_kernels.gemm import mxfp8_n5120_k8192 as wo_b
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.reference import G, quantize_mxfp8
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.reference import K as KA
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.reference import N as NA

COUNT = 100
N, K = wo_b.N, wo_b.K


def capture(fn):
    fn(0)
    fn(COUNT - 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
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
    ap.add_argument("--ms", default="769,800,848,896,960,1024")
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    wa_codes, wa_scale = quantize_mxfp8((torch.randn(G * NA, KA, generator=gen, device="cuda") * 0.05).bfloat16())
    wa_exp = wa_scale.view(G * NA, KA // 32)[::32].reshape(G, NA // 32, KA // 32).contiguous()
    wa = shuffle_weight(wa_codes.view(torch.float8_e4m3fn).view(G * NA, KA), (16, 16)).view(G * NA, KA)
    codes, scale = quantize_mxfp8((torch.randn(N, K, generator=gen, device="cuda") * 0.05).bfloat16())
    blk = scale[::32].contiguous()
    wb = shuffle_weight(codes.view(torch.float8_e4m3fn).view(N, K), (16, 16))
    wbs = wo_b.preshuffle_weight_scale(blk)
    grouped.warmup_grouped_gemm(torch.device("cuda"))
    wo_b.warmup_mxfp8_linear(wb, wbs)
    copies = -(-(500 << 20) // (N * K)) + 1
    wbs_copies = [(wb.clone(), wbs.clone(), blk.clone()) for _ in range(copies)]
    wa_copies = [(wa.clone(), wa_exp.clone()) for _ in range(16)]
    for m in (int(v) for v in args.ms.split(",")):
        xs = [quantize_mxfp8(torch.randn(m, G * KA, generator=gen, device="cuda").bfloat16()) for _ in range(COUNT)]

        def served(j):
            a, e = wa_copies[j % 16]
            w, _, b = wbs_copies[j % copies]
            c, s = grouped.grouped_gemm_mxfp8(xs[j][0], xs[j][1], a, e)
            return gemm_a8w8_blockscale_bpreshuffle(
                c, w, s.view(torch.float8_e8m0fnu), b.view(torch.float8_e8m0fnu), torch.bfloat16
            )

        def ours(j):
            a, e = wa_copies[j % 16]
            w, s, _ = wbs_copies[j % copies]
            return wo_b.mxfp8_linear(grouped.grouped_gemm(xs[j][0], xs[j][1], a, e), w, s)

        ref, out = served(0), ours(0)
        rel = ((ref.float() - out.float()).norm() / ref.float().norm()).item()
        graphs = {"served": capture(served), "ours": capture(ours)}
        for g in graphs.values():
            for _ in range(5):
                g.replay()
        samples = {k: [] for k in graphs}
        for i in range(9):
            order = list(graphs) if i % 2 == 0 else list(graphs)[::-1]
            for k in order + order[::-1]:
                samples[k].append(replay_us(graphs[k]))
        row = {k: statistics.median(v) for k, v in samples.items()}
        print(f"M {m}: served pair {row['served']:.2f} us, wo_a bf16 + this entry {row['ours']:.2f} us, rel L2 {rel:.1e}")
        del graphs, xs


if __name__ == "__main__":
    with torch.inference_mode():
        main()
