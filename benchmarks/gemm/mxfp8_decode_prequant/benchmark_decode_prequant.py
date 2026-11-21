# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the decode GEMM's forms at N 4608 x K 5120 (the gate_up weight): row-major pre-quantized and
bf16 activations, the split-K sum in a separate launch or in the last workgroup of a tile, and the swiglu + MXFP8
epilogue against the plain bf16 GEMM of the same row. Random weights in cold copies (> 400 MB), 100 distinct inputs,
5 rounds of 5 replays, median us per call.

  python3 benchmarks/gemm/mxfp8_decode_prequant/benchmark_decode_prequant.py
"""

import statistics

import torch

from amd_kernels.gemm.mxfp8_common import gemm as common
from amd_kernels.gemm.mxfp8_decode_prequant import Config, gemm, gemm_swiglu_mxfp8
from amd_kernels.gemm.mxfp8_decode_prequant.reference import quantize_mxfp8

N, K, COUNT = 4608, 5120, 100
# (rows, tile config of the plain form, split-K config of the fused / unfused comparison)
ROWS = [
    (16, Config(1, 2, 1), Config(1, 4, 1, 5, 0, None, 1)),
    (32, Config(1, 3, 1), Config(1, 4, 1, 5, 0, None, 1)),
    (64, Config(2, 3, 1), Config(2, 4, 1, 5, 0, None, 1)),
    (128, Config(4, 3, 1), Config(4, 4, 1, 5, 0, None, 1)),
]


def graph_us(call):
    call(0)
    call(COUNT - 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(COUNT):
            call(j)
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / (5 * COUNT))
    return statistics.median(times)


def main():
    gen = torch.Generator(device="cuda").manual_seed(0)
    q, s = quantize_mxfp8((torch.randn(N, K, generator=gen, device="cuda") * 0.05).bfloat16())
    w, ws = common.preshuffle_mxfp8_weight(q, s)
    copies = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(-(-(400 << 20) // (N * K)) - 1)]
    print("rows | rm | bf16 in | swiglu rm | swiglu bf16 | split-K sum in its own launch | in the last workgroup")
    for m, plain, sk in ROWS:
        xs = [torch.randn(m, K, generator=gen, device="cuda").bfloat16() for _ in range(COUNT)]
        qs = [quantize_mxfp8(x) for x in xs]

        def call(kind, cfg):
            def run(j):
                wj, wsj = copies[j % len(copies)]
                if kind == "rm":
                    return gemm(qs[j][0], qs[j][1], wj, wsj, cfg, key="bench")
                if kind == "bf16":
                    return gemm(xs[j], None, wj, wsj, cfg, key="bench")
                if kind == "swiglu rm":
                    return gemm_swiglu_mxfp8(qs[j][0], qs[j][1], wj, wsj, cfg, key="bench")
                return gemm_swiglu_mxfp8(xs[j], None, wj, wsj, cfg, key="bench")
            return run

        unfused = Config(sk.rt, sk.ni, sk.kw, sk.ks)
        row = [
            graph_us(call("rm", plain)), graph_us(call("bf16", plain)),
            graph_us(call("swiglu rm", sk)), graph_us(call("swiglu bf16", sk)),
            graph_us(call("rm", unfused)), graph_us(call("rm", sk)),
        ]  # fmt: skip
        print(f"{m:4d} | " + " | ".join(f"{t:.2f}" for t in row))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
