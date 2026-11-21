# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of bf16_linear against the stock op (vLLM's rocm_unquantized_gemm_impl when importable, else
F.linear) per served shape and row count: one graph of calls over distinct weight copies totalling more than 768 MB
(more than the 256 MB MALL: HBM on every call, as in serving), 8 replays x 7 rounds, minimum us per call.
Roofline: bytes = weight + x + y; floor = empty graph kernel 1.57 us + bytes / 6.9 TB/s (measured nontemporal HBM
read); %floor = floor / measured.
Run: python3 benchmarks/gemm/bf16_skinny_common/benchmark_linear.py [--shapes N,K ...] [--rows M ...]
"""
import argparse
import math
import os

import torch

from amd_kernels.gemm.bf16_skinny_common import SHAPES, bf16_linear, pack_weight, warmup_bf16_linear

HBM_TBPS, LAUNCH_US = 6.9, 1.57
STREAM = torch.cuda.Stream()  # the stock skinny GEMM keeps a capture workspace per capturing stream (8 slots)


def timing(fn, copies, iters=8, rounds=7):
    stream = STREAM
    with torch.cuda.stream(stream):
        fn()
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / iters / copies)
    return min(times)


def stock_op():
    os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
    try:
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl

        return lambda x, w: rocm_unquantized_gemm_impl(x, w, None), "vLLM rocm_unquantized_gemm_impl"
    except Exception:  # noqa: BLE001
        return (lambda x, w: torch.nn.functional.linear(x, w)), "F.linear"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", nargs="*", default=None)
    p.add_argument("--rows", type=int, nargs="*", default=[12, 24, 40, 64, 80, 96, 128, 160, 192])
    a = p.parse_args()
    shapes = [tuple(int(v) for v in s.split(",")) for s in a.shapes] if a.shapes else SHAPES
    stock, stock_name = stock_op()
    print(f"stock = {stock_name}; floor = {LAUNCH_US} us + bytes / {HBM_TBPS} TB/s")
    print("N      K      M    bytes MB  stock us  this us  speedup  this TB/s  % of floor")
    for n, k in shapes:
        copies = max(2, min(400, math.ceil(768e6 / (n * k * 2))))
        torch.manual_seed(0)
        ws = [(torch.randn(n, k, device="cuda") * 0.02).bfloat16() for _ in range(copies)]
        wps = [pack_weight(w) for w in ws]
        warmup_bf16_linear(wps[0])
        for m in a.rows:
            x = torch.randn(m, k, device="cuda").bfloat16()
            y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            def old():
                for w in ws:
                    stock(x, w)

            def new():
                for wp in wps:
                    bf16_linear(x, wp, y)
            t_old, t_new = timing(old, copies), timing(new, copies)
            byt = n * k * 2 + m * k * 2 + m * n * 2
            floor = LAUNCH_US + byt / (HBM_TBPS * 1e6)
            print(f"{n:6d} {k:6d} {m:4d} {byt/1e6:9.1f} {t_old:9.2f} {t_new:8.2f} {t_old/t_new:7.2f}x "
                  f"{byt/t_new/1e6:9.2f} {100*floor/t_new:9.0f}%", flush=True)
        del ws, wps
