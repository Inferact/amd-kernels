# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``clamp_silu_mul_mxfp8`` per row count against the served path (``SiluAndMulWithClamp``'s HIP
op + ``mxfp8_e4m3_quantize``: two launches): 100 distinct [m, 4608] gate_up inputs per graph (bigger than the LLC),
median of 5 rounds x 5 replays, us per call, rounds A B B A.
Run: python3 benchmarks/activation/clamp_silu_mul_mxfp8_i2304/benchmark_clamp_silu_mul.py
"""

import statistics

import torch
from amd_kernels.activation.clamp_silu_mul_mxfp8_i2304 import clamp_silu_mul_mxfp8, warmup_clamp_silu_mul_mxfp8

NIN = 100
LIMIT = 10.0
ROWS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768)


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


def abba(graphs):
    names = list(graphs)
    order = names + names[::-1]
    res = {n: [] for n in names}
    for n in order:
        res[n].append(timed(graphs[n]))
    return {n: statistics.median(v) for n, v in res.items()}


def main():
    from vllm import _custom_ops  # noqa: F401  (registers the HIP op)
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    warmup_clamp_silu_mul_mxfp8()
    print("rows | served silu_and_mul_with_clamp + quant | this entry")
    for m in ROWS:
        xs = [(torch.randn(m, 4608, device="cuda") * 4).bfloat16() for _ in range(NIN)]
        outs = [
            (
                torch.empty(m, 2304, device="cuda", dtype=torch.float8_e4m3fn),
                torch.empty(m, 72, device="cuda", dtype=torch.uint8),
            )
            for _ in range(NIN)
        ]
        acts = torch.empty(m, 2304, device="cuda", dtype=torch.bfloat16)

        def mine():
            for x, (c, s) in zip(xs, outs):
                clamp_silu_mul_mxfp8(x, LIMIT, codes=c, scale=s)

        def base():
            for x in xs:
                torch.ops._C.silu_and_mul_with_clamp(acts, x, LIMIT, 1.0, 0.0)
                mxfp8_e4m3_quantize(acts)

        t = abba({"base": graph(base), "this": graph(mine)})
        print(f"{m} | {t['base']:.2f} | {t['this']:.2f}", flush=True)
        del xs, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
