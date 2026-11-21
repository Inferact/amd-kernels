# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the N 4608 x K 5120 MXFP8 GEMM entries per row count, A B B A against the served path.

Random E4M3 weights with 32 x 32 block scales in cold copies (> 400 MB), 100 distinct inputs per graph, 9 rounds (each
A B then B A) of 5 replays, median us per call. Served path: vLLM's Triton MXFP8 quantizer, then AITER's FlyDSL
``gemm_a8w8_blockscale_bpreshuffle`` on the same preshuffled weight and the block scales (then this repo's clamped SiLU-mul + MXFP8 kernel for the gate_up chains);
pre-quantized scenarios start from the quantized activation. Run it with AITER's PR #6605 on ``PYTHONPATH`` for that
column; ``--baseline-only`` times the served path alone.

  python3 benchmarks/gemm/mxfp8_n4608_k5120/benchmark_mxfp8_n4608_k5120.py [--ms 16,64] [--scenarios linear] [--out result.json]
"""

import argparse
import json
import statistics

import torch

from amd_kernels.gemm.mxfp8_n4608_k5120 import K, N, kernel, mxfp8_linear, preshuffle_weight, gate_up_swiglu_mxfp8, gate_up_swiglu_mxfp8_prequant
from amd_kernels.gemm.mxfp8_n4608_k5120.reference import COS_MIN, compare, dequantize_mxfp8, quantize_mxfp8, SWIGLU_COS_MIN, SWIGLU_REL_L2_MAX

COUNT = 100
COLD_BYTES = 400 << 20


def base_prequant(q, s, wp, blk):
    from aiter import gemm_a8w8_blockscale_bpreshuffle

    out = torch.empty(q.shape[0], N, dtype=torch.bfloat16, device=q.device)
    gemm_a8w8_blockscale_bpreshuffle(q, wp, s.view(torch.float8_e8m0fnu), blk.view(torch.float8_e8m0fnu),
                                     torch.bfloat16, out=out)
    return out


def base_linear(x, wp, blk):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    q, s = mxfp8_e4m3_quantize(x)
    return base_prequant(q, s, wp, blk)


def base_swiglu_prequant(q, s, wp, blk):
    """AITER GEMM, then the clamped SiLU-mul + MXFP8 launch."""
    from amd_kernels.activation.clamp_silu_mul_mxfp8_i2304 import clamp_silu_mul_mxfp8

    return clamp_silu_mul_mxfp8(base_prequant(q, s, wp, blk), 10.0)


def base_swiglu(x, wp, blk):
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    q, s = mxfp8_e4m3_quantize(x)
    return base_swiglu_prequant(q, s, wp, blk)


SCENARIOS = {
    "linear": (kernel.LINEAR, lambda x, q, s, wp, ws, blk: mxfp8_linear(x, wp, ws), lambda x, q, s, wp, ws, blk: base_linear(x, wp, blk)),
    "swiglu": (kernel.SWIGLU, lambda x, q, s, wp, ws, blk: gate_up_swiglu_mxfp8(x, wp, ws), lambda x, q, s, wp, ws, blk: base_swiglu(x, wp, blk)),
    "swiglu_prequant": (kernel.SWIGLU_PREQUANT, lambda x, q, s, wp, ws, blk: gate_up_swiglu_mxfp8_prequant(q, s, wp, ws), lambda x, q, s, wp, ws, blk: base_swiglu_prequant(q, s, wp, blk)),
}


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


def similar(a, b):
    """The two paths agree: bf16 outputs, or MXFP8 (codes, scales) pairs, within the contracts."""
    if isinstance(a, tuple):
        r = compare(dequantize_mxfp8(*a), dequantize_mxfp8(*b))
        return r["cos"] >= SWIGLU_COS_MIN
    return compare(a, b)["cos"] >= COS_MIN


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--baseline-only", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    codes = torch.randint(0, 120, (N, K), generator=gen, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    blk = torch.randint(118, 128, (N // 32, K // 32), generator=gen, device="cuda", dtype=torch.uint8)
    wp, ws = preshuffle_weight(codes, blk)
    n_copies = -(-COLD_BYTES // (N * K))
    copies = [(wp, ws, blk)] + [(wp.clone(), ws.clone(), blk.clone()) for _ in range(n_copies - 1)]
    result = {}
    for name in args.scenarios.split(","):
        table, new_fn, base_fn = SCENARIOS[name]
        ms = [int(m) for m in args.ms.split(",")] if args.ms else [row[1] for row in table]
        result[name] = {}
        for m in ms:
            xs = [torch.randn(m, K, generator=gen, device="cuda").bfloat16() for _ in range(COUNT)]
            qs = [quantize_mxfp8(x) for x in xs]
            qs = [(c, e) for c, e in qs]

            def call(fn, j):
                wpj, wsj, bj = copies[j % len(copies)]
                return fn(xs[j], qs[j][0], qs[j][1], wpj, wsj, bj)

            graphs = {"base": capture(lambda j: call(base_fn, j))}
            if not args.baseline_only:
                assert similar(call(new_fn, 0), call(base_fn, 0)), (name, m)
                graphs["new"] = capture(lambda j: call(new_fn, j))
            for g in graphs.values():
                for _ in range(5):
                    g.replay()
            samples = {arm: [] for arm in graphs}
            for i in range(args.rounds):
                order = list(graphs) if i % 2 == 0 else list(graphs)[::-1]
                for arm in order + order[::-1]:
                    samples[arm].append(replay_us(graphs[arm]))
            row = {arm: statistics.median(v) for arm, v in samples.items()}
            row["min"] = {arm: min(v) for arm, v in samples.items()}
            row["ranges"] = {arm: [min(v), max(v)] for arm, v in samples.items()}
            result[name][m] = row
            print(name, m, json.dumps(row), flush=True)
            del graphs, xs, qs
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
