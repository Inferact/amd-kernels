# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sweeps AITER's FlyDSL a8w4 stage 1 / stage 2 kernels per token count and prints the fastest pair as the
``baseline.BASELINE`` rows: per sort block_m, the stage 1 kernel (fused MXFP8 requant) is timed with the default
stage 2, then the stage 2 kernels with the best stage 1 (``--deterministic`` keeps stage 2 ``reduce``, else
``atomic`` and ``reduce``)."""
import argparse
import json

import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import ExpertWeights
from tests.moe.mxfp4_h5120_i2304_e48_top6.workload import make_activation, make_experts, make_routes

from baseline import aiter_experts


def stage1_names(bm):
    from aiter.ops.flydsl.moe_kernels import get_flydsl_stage1_kernels

    return [n + "_fp8" for n, p in get_flydsl_stage1_kernels("fp8", "fp4", "bf16").items()
            if p["tile_m"] == bm and p["xcd_swizzle"] == 0 and p["waves_per_eu"] in (2, 3, 4) and p["k_batch"] == 1]


def stage2_names(bm, deterministic):
    from aiter.ops.flydsl.moe_kernels import get_flydsl_stage2_kernels

    modes = ("reduce",) if deterministic else ("atomic", "reduce")
    return [n for n, p in get_flydsl_stage2_kernels("fp8", "fp4", "bf16").items()
            if p["tile_m"] == bm and p["xcd_swizzle"] == 0 and p["mode"] in modes and not p.get("persist")]


def stage_us(call, inputs, name, iters=20):
    import aiter.fused_moe as fm

    captured = []
    fm.kernel_bench_callable = captured
    try:
        call(*inputs)
    finally:
        fm.kernel_bench_callable = None
    fn = dict(captured)[name]
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 32, 64, 128, 192, 384, 768])
    parser.add_argument("--dist", default="uniform", choices=["uniform", "skew"])
    parser.add_argument("--block-m", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    w = ExpertWeights.from_row_major(*make_experts())
    best = {}
    for n in args.tokens:
        x = make_activation(n, seed=n)
        ids, wt = make_routes(n, args.dist, seed=n)
        inputs = (x[0], x[1], wt, ids)
        rows = []
        for bm in args.block_m:
            d2 = f"flydsl_moe2_afp8_wfp4_bf16_t{bm}x128x256_reduce"
            t1 = []
            for k1 in stage1_names(bm):
                try:
                    t1.append((stage_us(aiter_experts(w, n, args.dist, (bm, k1, d2)), inputs, "stage1"), k1))
                except Exception as exc:
                    print("skip", k1, str(exc)[:60])
            if not t1:
                continue
            u1, k1 = min(t1)
            t2 = []
            for k2 in stage2_names(bm, args.deterministic):
                try:
                    t2.append((stage_us(aiter_experts(w, n, args.dist, (bm, k1, k2)), inputs, "stage2"), k2))
                except Exception as exc:
                    print("skip", k2, str(exc)[:60])
            u2, k2 = min(t2)
            rows.append((u1 + u2, bm, k1, k2, u1, u2))
            print(json.dumps({"tokens": n, "bm": bm, "stage1_us": u1, "stage2_us": u2, "k1": k1, "k2": k2}), flush=True)
        total, bm, k1, k2, _, _ = min(rows)
        best[n] = (bm, k1, k2)
        print(f"BASELINE[{n}] = {best[n]!r}  # stages {total:.1f} us", flush=True)
    print("BASELINE =", json.dumps(best))


if __name__ == "__main__":
    main()
