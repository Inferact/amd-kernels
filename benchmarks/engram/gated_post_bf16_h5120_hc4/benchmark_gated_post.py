# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``gated_post`` against vLLM's Triton ``_fused_engram_post_wkv_kernel`` (token mask, separate
output), per row count of a decode step: 100 distinct (hidden, kv) pairs per graph, together far above the LLC.
Median of 5 rounds x 5 replays, us per call; rounds alternate A B B A.
Run: python3 benchmarks/engram/gated_post_bf16_h5120_hc4/benchmark_gated_post.py [--rows 16 64 ...]
"""

import argparse
import statistics

import torch
from amd_kernels.engram.gated_post_bf16_h5120_hc4 import gated_post

H, HC = 5120, 4
NIN = 100


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
    ap.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64, 128, 192, 256, 384, 512, 768])
    args = ap.parse_args()
    dev = torch.device("cuda")
    try:
        from vllm.models.deepseek_v41.common.engram import _fused_engram_post_wkv_kernel as triton_post
        from vllm.triton_utils import triton
    except ImportError:
        triton_post = None
    gated_post(torch.zeros(1, HC, H, dtype=torch.bfloat16, device=dev), torch.zeros(1, (HC + 1) * H, dtype=torch.bfloat16, device=dev),
               torch.zeros(HC, H, dtype=torch.bfloat16, device=dev), torch.zeros(HC, H, dtype=torch.bfloat16, device=dev),
               torch.ones(1, dtype=torch.bool, device=dev))
    q = torch.randn(HC, H, device=dev).bfloat16()
    k = torch.randn(HC, H, device=dev).bfloat16()
    print("rows | this entry us | vLLM Triton us")
    for t in args.rows:
        hs = [torch.randn(t, HC, H, device=dev).bfloat16() for _ in range(NIN)]
        kvs = [torch.randn(t, (HC + 1) * H, device=dev).bfloat16() for _ in range(NIN)]
        mask = torch.rand(t, device=dev) > 0.1
        outs = [torch.empty_like(hs[0]) for _ in range(2)]

        def ours():
            for h, kv in zip(hs, kvs):
                gated_post(h, kv, q, k, mask, out=outs[0])

        def base():
            for h, kv in zip(hs, kvs):
                triton_post[(t * HC,)](
                    h, kv, q, k, mask, outs[1], t, h.stride(0), h.stride(1), h.stride(2), kv.stride(0), kv.stride(1),
                    q.stride(0), q.stride(1), k.stride(0), k.stride(1), mask.stride(0), outs[1].stride(0),
                    outs[1].stride(1), outs[1].stride(2), 1e-20, 1e-6, DIM=H, HC_MULT=HC,
                    BLOCK_SIZE=triton.next_power_of_2(H), HAS_MASK=True, num_warps=8)

        go = graph(ours)
        gb = graph(base) if triton_post is not None else None
        to, tb = [], []
        for _ in range(2):  # A B B A
            to.append(timed(go, NIN))
            if gb is not None:
                tb.append(timed(gb, NIN))
                tb.append(timed(gb, NIN))
            to.append(timed(go, NIN))
        print(f"{t} | {statistics.median(to):.1f} | {statistics.median(tb) if tb else float('nan'):.1f}", flush=True)
        del hs, kvs, go, gb
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
