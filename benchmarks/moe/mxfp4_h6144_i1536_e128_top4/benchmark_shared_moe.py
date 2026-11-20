# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Public routed A4W4 + shared A8W8 entry, synthetic weights, no collective; ``--baseline``: the vLLM TP2 block on the
same weights and inputs in the same process.

Each shape uses one live graph. An overlap claim also needs capture-order
controls; a checkpoint claim needs a same-buffer A/B on real weights.
"""
import argparse
import json
import os
import statistics
import sys

import torch
from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeWeights, SharedWeights, mxfp4_moe
from tests.moe.mxfp4_h6144_i1536_e128_top4.workload import make_activation, make_experts

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def weights():
    raw, _ = make_experts()
    generator = torch.Generator(device="cuda").manual_seed(607)

    def fp8(n, k):
        q = (torch.randn((n, k), device="cuda", generator=generator) * 20).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
        return q.view(torch.uint8), s

    shared = SharedWeights.from_row_major(*fp8(3072, 6144), *fp8(6144, 1536))
    gate = torch.randn((128, 6144), device="cuda", generator=generator) / 6144**.5
    bias = torch.randn((128,), device="cuda", generator=generator) * .01
    return MoeWeights.from_row_major(gate, bias, *raw, shared=shared)


def view_weights():
    """(MoeWeights over TP rank 1's shard copy, the same over the full copy read by view): one routed set."""
    from dataclasses import replace

    from tests.moe.mxfp4_h6144_i1536_e128_top4.workload import make_full_shared
    base = weights()
    raw, shards = make_full_shared()
    copy = replace(base, shared_weights=SharedWeights.from_row_major(*shards[1]))
    return copy, replace(base, shared_weights=SharedWeights.from_row_major(*raw, tp_rank=1))


def measure(w, m, args, block=None):
    copies = args.copies or (100 if m <= 2048 else 16)
    xs = [make_activation(m, seed=1000 * m + i) for i in range(copies)]

    def call(x):
        return mxfp4_moe(x, w, return_partials=args.partials) if block is None else block(x)

    def flat(y):
        return list(y) if isinstance(y, tuple) else [y]

    call(xs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [call(x) for x in xs]
    for _ in range(20):
        graph.replay()
        torch.cuda.synchronize()
    samples = []
    for _ in range(args.rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            if args.eager:
                for x in xs:
                    call(x)
            else:
                graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (args.replays * copies))
    for x, captured in zip(xs, outputs if block is None else ()):
        eager = call(x)
        torch.cuda.synchronize()
        assert all(torch.isfinite(t).all() and torch.equal(t, expected)
                   for t, expected in zip(flat(captured), flat(eager)))
    return {"tokens": m, "copies": copies, "median_us": statistics.median(samples),
            "range_us": [min(samples), max(samples)], "rounds_us": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--copies", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--partials", action="store_true")
    parser.add_argument("--eager", action="store_true",help="time real entry calls, including host issue, instead of graph replay")
    parser.add_argument("--baseline", action="store_true",
                        help="also time the block vLLM runs at TP2 without this entry (benchmark_baseline_moe.py)")
    parser.add_argument("--view", action="store_true",
                        help="the shard copy vs the full copy read by view (rank 1), alternating ``--ab`` times")
    parser.add_argument("--ab", type=int, default=3)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": props.name, "pci": props.pci_bus_id, "torch": torch.__version__,
                      "scope": "synthetic routed + shared, no collective", "partials": args.partials,
                      "mode": "eager" if args.eager else "graph"}))
    if args.view:
        copy, view = view_weights()
        for m in args.tokens:
            runs = {"copy": [], "view": []}
            for _ in range(args.ab):
                for name, w in (("copy", copy), ("view", view)):
                    runs[name].append(measure(w, m, args)["median_us"])
                    torch.cuda.empty_cache()
            rec = {"tokens": m, **{f"{k}_us": statistics.median(v) for k, v in runs.items()}, "runs": runs}
            rec["change"] = f"{rec['view_us'] / rec['copy_us'] - 1:+.2%}"
            print(json.dumps(rec), flush=True)
        return
    w = weights()
    if args.baseline:
        import benchmark_baseline_moe as base
        wb, side = base.weights(), torch.cuda.Stream()
    for m in args.tokens:
        rec = measure(w, m, args)
        if args.baseline:
            rec["baseline_us"] = measure(wb, m, args, block=lambda x: base.block(x, wb, side))["median_us"]
            rec["change"] = f"{rec['median_us'] / rec['baseline_us'] - 1:+.1%}"
        print(json.dumps(rec), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
