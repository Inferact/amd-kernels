# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Public routed A4W4 + shared A8W8 entry, synthetic weights, no collective.

Each shape uses one live graph. An overlap claim also needs capture-order
controls; a checkpoint claim needs a same-buffer A/B on real weights.
"""
import argparse
import json
import statistics

import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, SharedWeights, mxfp4_moe
from tests.moe.mxfp4_h6144_i768_e128_top4.workload import make_activation, make_experts


def weights():
    raw, _ = make_experts()
    generator = torch.Generator(device="cuda").manual_seed(607)

    def fp8(n, k):
        q = (torch.randn((n, k), device="cuda", generator=generator) * 20).to(torch.float8_e4m3fn)
        s = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=generator)
        return q.view(torch.uint8), s

    shared = SharedWeights.from_row_major(*fp8(1536, 6144), *fp8(6144, 768))
    gate = torch.randn((128, 6144), device="cuda", generator=generator) / 6144**.5
    bias = torch.randn((128,), device="cuda", generator=generator) * .01
    return MoeWeights.from_row_major(gate, bias, *raw, shared=shared)


def measure(w, m, args):
    copies = args.copies or (100 if m <= 2048 else 16)
    xs = [make_activation(m, seed=1000 * m + i) for i in range(copies)]

    def call(x):
        return mxfp4_moe(x, w, return_partials=args.partials)

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
    for x, captured in zip(xs, outputs):
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
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": props.name, "pci": props.pci_bus_id, "torch": torch.__version__,
                      "scope": "synthetic routed + shared, no collective", "partials": args.partials,
                      "mode": "eager" if args.eager else "graph"}))
    w = weights()
    for m in args.tokens:
        print(json.dumps(measure(w, m, args)), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
