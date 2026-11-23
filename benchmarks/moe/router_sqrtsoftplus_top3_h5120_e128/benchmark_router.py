# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``router_topk`` per row count against the served path (``fp32_router_gemm_dispatch`` +
``topk_hash_softplus_sqrt``: the gate GEMM and the top-k, two or more launches): 100 distinct [m, 5120] inputs per
graph (bigger than the LLC), median of 5 rounds x 5 replays, us per call, rounds A B B A.
Run: python3 benchmarks/moe/router_sqrtsoftplus_top3_h5120_e128/benchmark_router.py
"""

import statistics

import torch
from amd_kernels.moe.router_sqrtsoftplus_top3_h5120_e128 import router_topk, shuffle_gate, warmup_router_topk

NIN = 100
ROWS = (16, 40, 80, 160, 320, 640, 768)


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
    from vllm import _custom_ops as ops
    import vllm.model_executor.layers.fused_moe.router.gate_linear  # noqa: F401  (registers the gate GEMM op)

    warmup_router_topk()
    w = (torch.randn(128, 5120, device="cuda") * 0.02).bfloat16()
    bias = torch.randn(128, device="cuda") * 0.1
    ws = shuffle_gate(w)
    print("rows | served gate GEMM + topk | this entry | this entry, load-time gate layout (shuffle_gate)")
    for m in ROWS:
        xs = [torch.randn(m, 5120, device="cuda").bfloat16() for _ in range(NIN)]
        outs = [
            (
                torch.empty(m, 3, device="cuda"),
                torch.empty(m, 3, device="cuda", dtype=torch.int32),
                torch.empty(m, 3, device="cuda", dtype=torch.int32),
            )
            for _ in range(NIN)
        ]

        def mine():
            for x, (tw, ti, _) in zip(xs, outs):
                router_topk(x, w, bias, 1.5, topk_weights=tw, topk_ids=ti)

        def mine_shuffled():
            for x, (tw, ti, _) in zip(xs, outs):
                router_topk(x, ws, bias, 1.5, topk_weights=tw, topk_ids=ti, weight_shuffled=True)

        def base():
            for x, (tw, ti, te) in zip(xs, outs):
                lg = torch.ops.vllm.fp32_router_gemm_dispatch(x, w, False)
                ops.topk_hash_softplus_sqrt(tw, ti, te, lg, True, 1.5, bias, None, None, None, None, 0)

        t = abba({"base": graph(base), "this": graph(mine), "shuf": graph(mine_shuffled)})
        print(f"{m} | {t['base']:.2f} | {t['this']:.2f} | {t['shuf']:.2f}", flush=True)
        del xs, outs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
