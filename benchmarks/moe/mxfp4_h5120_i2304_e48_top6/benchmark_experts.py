# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routed experts, graph replay over distinct inputs, ``mxfp4_experts`` vs AITER's FlyDSL a8w4 ``fused_moe``.

One graph holds ``--copies`` calls on distinct activations and routes; the weights (0.9 GB, beyond the LLC) are
shared. The rounds alternate A B B A (this entry, baseline, baseline, this entry); medians are us per call.
"""
import argparse
import json
import statistics

import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6 import ExpertWeights, mxfp4_experts
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6.reference import E, H, I
from tests.moe.mxfp4_h5120_i2304_e48_top6.workload import TOKENS, make_activation, make_experts, make_routes

from baseline import aiter_experts


def graph_of(call, inputs):
    call(inputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [call(x) for x in inputs]
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, outs


def replay_us(graph, copies, replays):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (replays * copies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=list(TOKENS))
    parser.add_argument("--dist", default="uniform", choices=["uniform", "skew"])
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3, help="A B B A passes")
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()
    raw = make_experts()
    w = ExpertWeights.from_row_major(*raw)
    del raw
    results = []
    for n in args.tokens:
        routes = [make_routes(n, args.dist, seed=1000 * n + i) for i in range(args.copies)]
        acts = [make_activation(n, seed=1000 * n + i) for i in range(args.copies)]
        inputs = list(zip(acts, routes))
        ours, ours_out = graph_of(lambda a: mxfp4_experts(a[0][0], a[0][1], w, a[1][1], a[1][0]), inputs)
        row = {"tokens": n, "dist": args.dist, "copies": args.copies}
        samples = {"this": []}
        order = ["this"]
        if not args.no_baseline:
            base_call = aiter_experts(w, n, args.dist)
            base, base_out = graph_of(lambda a: base_call(a[0][0], a[0][1], a[1][1], a[1][0]), inputs)
            row["agreement_cos"] = torch.nn.functional.cosine_similarity(
                ours_out[0].double().flatten(), base_out[0].double().flatten(), dim=0).item()
            samples["baseline"] = []
            order = ["this", "baseline", "baseline", "this"]
        graphs = {"this": ours, "baseline": base if not args.no_baseline else None}
        for _ in range(args.rounds):
            for name in order:
                samples[name].append(replay_us(graphs[name], args.copies, args.replays))
        for name, s in samples.items():
            row[name + "_us"] = statistics.median(s)
            row[name + "_rounds"] = s
        results.append(row)
        print(json.dumps(row), flush=True)
    return results


if __name__ == "__main__":
    main()
