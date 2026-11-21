# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``grouped_gemm`` and ``grouped_gemm_mxfp8`` per row count against AITER's FlyDSL
``batched_gemm_a8w8_mxscale_bpreshuffle`` (and, for the MXFP8 entry, that plus vLLM's ``mxfp8_e4m3_quantize``, the
quantize launch the epilogue removes). The weight rotates through 16 cold copies (> 500 MB), 100 distinct inputs per
graph; rounds alternate baseline, ours, ours, baseline (A B B A); median of 5 rounds x 5 replays, us per call.
Run: python3 benchmarks/gemm/mxfp8_grouped_g8_n1024_k4096/benchmark_grouped_gemm.py [--rows 16,64,...]
"""

import argparse
import statistics

import torch
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8_mxscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096 import grouped_gemm, grouped_gemm_mxfp8, warmup_grouped_gemm
from amd_kernels.gemm.mxfp8_grouped_g8_n1024_k4096.reference import BLOCK, G, K, N

NIN = 100
NW = 16
ROWS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768)


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


def weights(dev):
    gen = torch.Generator(device=dev).manual_seed(1)
    base = torch.randn(G * N, K, device=dev, generator=gen).to(torch.float8_e4m3fn).view(torch.uint8)
    out = []
    for i in range(NW):
        w = shuffle_weight(torch.roll(base, 16 * i, 0).view(torch.float8_e4m3fn), (16, 16)).view(G, N, K).clone()
        s = torch.randint(118, 126, (G, N // BLOCK, K // BLOCK), dtype=torch.uint8, device=dev, generator=gen)
        out.append((w, s))
    return out


def inputs(m, dev):
    gen = torch.Generator(device=dev).manual_seed(m)
    xs = [torch.randn(m, G, K, device=dev, generator=gen).to(torch.float8_e4m3fn) for _ in range(NIN)]
    ss = [torch.randint(118, 126, (m, G, K // BLOCK), dtype=torch.uint8, device=dev, generator=gen) for _ in range(NIN)]
    return xs, ss


def ab_ba(ga, gb):
    a, b = [], []
    for _ in range(2):
        a.append(timed(ga))
        b.append(timed(gb))
        b.append(timed(gb))
        a.append(timed(ga))
    return statistics.median(a), statistics.median(b)


def main():
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    args = ap.parse_args()
    dev = torch.device("cuda", 0)
    warmup_grouped_gemm(dev)
    ws = weights(dev)
    print("rows | AITER bf16 us | grouped_gemm us | AITER + quant us | grouped_gemm_mxfp8 us", flush=True)
    for m in map(int, args.rows.split(",")):
        xs, ss = inputs(m, dev)

        def base():
            for i in range(NIN):
                w, s = ws[i % NW]
                batched_gemm_a8w8_mxscale_bpreshuffle(xs[i], w, ss[i], s, torch.bfloat16)

        def base_quant():
            for i in range(NIN):
                w, s = ws[i % NW]
                out = batched_gemm_a8w8_mxscale_bpreshuffle(xs[i], w, ss[i], s, torch.bfloat16)
                mxfp8_e4m3_quantize(out.view(m, -1))

        def ours():
            for i in range(NIN):
                w, s = ws[i % NW]
                grouped_gemm(xs[i].view(m, -1), ss[i].view(m, -1), w.view(G * N, K), s)

        def ours_quant():
            for i in range(NIN):
                w, s = ws[i % NW]
                grouped_gemm_mxfp8(xs[i].view(m, -1), ss[i].view(m, -1), w.view(G * N, K), s)

        gb, go, gbq, goq = graph(base), graph(ours), graph(base_quant), graph(ours_quant)
        b, o = ab_ba(gb, go)
        bq, oq = ab_ba(gbq, goq)
        print(f"{m} | {b:.2f} | {o:.2f} | {bq:.2f} | {oq:.2f}", flush=True)
        del gb, go, gbq, goq, xs, ss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
