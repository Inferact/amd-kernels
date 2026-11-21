# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Finds AITER's fastest FlyDSL a8w4 ``fused_moe`` kernels per token count for the baseline (``baseline.BASELINE``):
per sort block ``bm``, the stage-1 kernels (fused MXFP8 requant) against the default stage 2, then the stage-2 kernels
against the best stage 1; graph-replay time of the whole call over distinct inputs. Prints the fastest rows.
Run: python3 benchmarks/moe/mxfp4_h5120_i2304_e16_top3/sweep_baseline.py --tokens 16 40 80 160 320 640 1024
"""
import argparse
import json
import statistics

import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e16_top3 import ExpertWeights
from tests.moe.mxfp4_h5120_i2304_e16_top3.workload import make_activation, make_experts, make_routes

from baseline import DEFAULT, aiter_experts

COPIES = 20


def graph_us(call, inputs):
    call(*inputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x in inputs:
            call(*x)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(4):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / (4 * len(inputs)))
    return statistics.median(times)


def candidates(bm):
    from aiter.ops.flydsl import moe_kernels as mk

    k1 = [k + "_fp8" for k in mk.get_flydsl_stage1_kernels("fp8", "fp4", "bf16")
          if f"_t{bm}x" in k and "xcd4" not in k and "kw" not in k]
    k2 = [k for k in mk.get_flydsl_stage2_kernels("fp8", "fp4", "bf16") if f"_t{bm}x" in k and "xcd4" not in k
          and "bnt2" not in k and "persist" not in k]
    return k1, k2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens", type=int, nargs="+", default=[16, 40, 80, 160, 320, 640, 1024])
    ap.add_argument("--bm", type=int, nargs="+", default=[16, 32, 64, 128])
    args = ap.parse_args()
    w = ExpertWeights.from_row_major(*make_experts())
    for n in args.tokens:
        inputs = [(make_activation(n, seed=100 * n + i)[0], make_activation(n, seed=100 * n + i)[1],
                   make_routes(n, "uniform", seed=100 * n + i)[1], make_routes(n, "uniform", seed=100 * n + i)[0])
                  for i in range(COPIES)]
        rows = []
        for bm in args.bm:
            k1s, k2s = candidates(bm)
            if not k1s or not k2s:
                continue
            d1, d2 = DEFAULT[1], DEFAULT[2]
            best1 = (1e9, None)
            for k1 in k1s:
                try:
                    us = graph_us(aiter_experts(w, n, (bm, k1, d2.replace("t32x", f"t{bm}x"))), inputs)
                except Exception:
                    continue
                best1 = min(best1, (us, k1))
            if best1[1] is None:
                continue
            best2 = (1e9, None)
            for k2 in k2s:
                try:
                    us = graph_us(aiter_experts(w, n, (bm, best1[1], k2)), inputs)
                except Exception:
                    continue
                best2 = min(best2, (us, k2))
            rows.append((best2[0], bm, best1[1], best2[1]))
        rows.sort()
        print(json.dumps({"tokens": n, "best": rows[:3]}), flush=True)


if __name__ == "__main__":
    main()
