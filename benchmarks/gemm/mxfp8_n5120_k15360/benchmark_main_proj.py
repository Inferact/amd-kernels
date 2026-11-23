# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``main_proj`` against vLLM's ``rocm_mxfp8_block32_gemm_quant`` (the ROCm MXFP8 linear of
32 x 32 block-scaled weights: AITER's preshuffled FlyDSL GEMM, or the Triton packed kernel with the quantizer fused
at small M) per row count. Weights in rotating copies (> 400 MB, pointer-distinct), distinct activations per graph
slot; rounds A B B A, median of 5 rounds x 5 replays, us per call.
Run: python3 benchmarks/gemm/mxfp8_n5120_k15360/benchmark_main_proj.py [--rows 160 320 640]
"""

import argparse
import statistics

import torch
from amd_kernels.gemm.mxfp8_n5120_k15360 import K, MainProjWeights, N, main_proj, warmup_main_proj
from amd_kernels.gemm.mxfp8_n5120_k15360.reference import quantize_mxfp8
from vllm.model_executor.kernels.linear.mxfp8.rocm_block32_gemm import rocm_mxfp8_block32_gemm_quant

ROWS = (32, 64, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024, 2048, 4096, 8192)


def timed(g, n):
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * n))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=list(ROWS))
    args = ap.parse_args()
    from aiter.ops.shuffle import shuffle_weight

    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(0)
    codes, _ = quantize_mxfp8((torch.randn(N, K, generator=gen, device=dev) * 0.05).bfloat16())
    block = torch.randint(118, 126, (N // 32, K // 32), generator=gen, device=dev, dtype=torch.uint8)
    copies = -(-(450 << 20) // (N * K))
    ours = [MainProjWeights.from_block_scales(codes.clone(), block.clone()) for _ in range(copies)]
    theirs = [(shuffle_weight(codes.clone(), (16, 16)), block.clone()) for _ in range(copies)]
    warmup_main_proj(ours[0])
    print("rows | baseline us | this entry us | change")
    for m in args.rows:
        n = 20 if m <= 1024 else 4
        xs = [(torch.randn(m, K, generator=gen, device=dev)).bfloat16() for _ in range(n)]

        def run_ours():
            for i, x in enumerate(xs):
                main_proj(x, ours[i % copies])

        def run_base():
            for i, x in enumerate(xs):
                wp, sc = theirs[i % copies]
                rocm_mxfp8_block32_gemm_quant(x, wp, sc, torch.bfloat16, wp)

        go, gb = graph(run_ours), graph(run_base)
        t_o, t_b = [], []
        for g in (go, gb, gb, go):
            (t_o if g is go else t_b).append(timed(g, n))
        o, b = statistics.median(t_o), statistics.median(t_b)
        print(f"{m} | {b:.1f} | {o:.1f} | {(o / b - 1) * 100:+.1f}% (A B B A: ours {t_o[0]:.1f}/{t_o[1]:.1f}, "
              f"base {t_b[0]:.1f}/{t_b[1]:.1f})", flush=True)
        del go, gb, xs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
