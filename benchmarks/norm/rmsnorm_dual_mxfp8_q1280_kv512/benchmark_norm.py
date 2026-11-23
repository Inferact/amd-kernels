# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``rmsnorm_dual_mxfp8`` per row count against the served Triton path (``fused_q_kv_rmsnorm`` +
``mxfp8_e4m3_quantize`` of q: two launches) and AITER's ``fused_dual_rmsnorm_mxfp8_quant`` (one Triton launch): 100
distinct [m, 1792] inputs per graph (bigger than the LLC), median of 5 rounds x 5 replays, us per call, rounds
A B B A. Run: python3 benchmarks/norm/rmsnorm_dual_mxfp8_q1280_kv512/benchmark_norm.py
"""

import statistics

import torch
from amd_kernels.norm.rmsnorm_dual_mxfp8_q1280_kv512 import rmsnorm_dual_mxfp8, warmup_rmsnorm_dual_mxfp8

NIN = 100
EPS = 1e-20
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
    """Median per graph of rounds in the order A B B A (each graph is timed twice)."""
    names = list(graphs)
    order = names + names[::-1]
    res = {n: [] for n in names}
    for n in order:
        res[n].append(timed(graphs[n]))
    return {n: statistics.median(v) for n, v in res.items()}


def main():
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
    from vllm.models.common.ops import fused_q_kv_rmsnorm

    try:
        from aiter.ops.triton.quant.fused_mxfp8_quant import fused_dual_rmsnorm_mxfp8_quant as aiter_fused
    except ImportError:
        aiter_fused = None
    warmup_rmsnorm_dual_mxfp8()
    qw = (torch.rand(1280, device="cuda") + 0.5).bfloat16()
    kw = (torch.rand(512, device="cuda") + 0.5).bfloat16()
    print("rows | triton norm + quant | aiter fused | this entry")
    for m in ROWS:
        xs = [(torch.randn(m, 1792, device="cuda") * 2).bfloat16() for _ in range(NIN)]
        splits = [x.split([1280, 512], dim=-1) for x in xs]
        outs = [
            (
                torch.empty(m, 1280, device="cuda", dtype=torch.float8_e4m3fn),
                torch.empty(m, 40, device="cuda", dtype=torch.uint8),
                torch.empty(m, 512, device="cuda", dtype=torch.bfloat16),
            )
            for _ in range(NIN)
        ]

        def mine():
            for (q, kv), (c, s, k) in zip(splits, outs):
                rmsnorm_dual_mxfp8(q, kv, qw, kw, EPS, codes=c, scale=s, kv_out=k)

        def base():
            for q, kv in splits:
                nq, nkv = fused_q_kv_rmsnorm(q, kv, qw, kw, EPS)
                mxfp8_e4m3_quantize(nq)

        def aiter():
            for (q, kv), (c, s, k) in zip(splits, outs):
                aiter_fused(q, kv, qw, kw, EPS, yq=c, sq=s, yk=k)

        graphs = {"base": graph(base), "this": graph(mine)}
        if aiter_fused is not None:
            graphs["aiter"] = graph(aiter)
        t = abba(graphs)
        print(f"{m} | {t['base']:.2f} | {t.get('aiter', float('nan')):.2f} | {t['this']:.2f}", flush=True)
        del graphs, xs, splits, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
