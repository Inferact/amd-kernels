# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``rmsnorm_mxfp8`` per row count against the served pair: vLLM's ``rms_norm`` op (bf16 out) and
its Triton ``mxfp8_e4m3_quantize`` (two launches): up to 100 distinct [m, 5120] inputs per graph (1 GB at most),
median of 5 rounds x 5 replays, us per call, rounds A B B A.
Run: python3 benchmarks/norm/rmsnorm_mxfp8_h5120/benchmark_rmsnorm.py
"""

import statistics

import torch
from amd_kernels.norm.rmsnorm_mxfp8_h5120 import rmsnorm_mxfp8, warmup_rmsnorm_mxfp8
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

ROWS = (16, 40, 80, 160, 192, 320, 384, 640, 768, 1024, 4096, 8192)
EPS = 1e-20
H = 5120


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


def main():
    warmup_rmsnorm_mxfp8()
    w = (torch.rand(H, device="cuda") + 0.5).bfloat16()
    print("rows | rms_norm + quantize us | this entry us | change")
    for m in ROWS:
        n = max(4, min(100, (1 << 30) // (m * H * 2)))
        xs = [torch.randn(m, H, device="cuda").bfloat16() for _ in range(n)]
        outs = [(torch.empty(m, H, dtype=torch.float8_e4m3fn, device="cuda"), torch.empty(m, H // 32, dtype=torch.uint8,
                                                                                             device="cuda")) for _ in range(n)]
        normed = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")

        def mine():
            for x, (c, s) in zip(xs, outs):
                rmsnorm_mxfp8(x, w, EPS, codes=c, scale=s)

        def base():
            for x in xs:
                torch.ops._C.rms_norm(normed, x, w, EPS)
                mxfp8_e4m3_quantize(normed)

        gm, gb = graph(mine), graph(base)
        tm, tb = [], []
        for g in (gm, gb, gb, gm):
            (tm if g is gm else tb).append(timed(g, n))
        a, b = statistics.median(tm), statistics.median(tb)
        print(f"{m} | {b:.2f} | {a:.2f} | {(a / b - 1) * 100:+.1f}%", flush=True)
        del gm, gb, xs, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
