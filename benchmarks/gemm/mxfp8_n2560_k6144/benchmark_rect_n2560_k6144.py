# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Cold-weight QKV 128x160 versus the previous 128x256 row, GEMM and insert chain.

Layer 39 rank 0; 26 weight copies (>400 MB); 100 seeded inputs; 3 x 5
alternated rounds of 10 graph replays. Eager entry calls are timed too.

  python3 benchmarks/gemm/mxfp8_n2560_k6144/benchmark_rect_n2560_k6144.py --checkpoint /path/to/model
"""
import argparse
import json
import statistics
from pathlib import Path

import torch
from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import norm_rope_nvfp4_insert
from benchmark_qkv_insert import inputs, measure, weights

OLD = g._Config(128, 256, 128, 1, 1, 1, "4wave", 2, 2)
COUNT = 100


def capture(call, *, pool=None):
    call(0)
    call(COUNT - 1)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        for j in range(COUNT):
            call(j)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def eager(call):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for j in range(COUNT):
        call(j)
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / COUNT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--ms", default="1537,1601,1792,1921,2048")
    ap.add_argument("--units", default="gemm,chain")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    copies, norms = weights(args.checkpoint, args.layer)
    result = {}
    for m in map(int, args.ms.split(",")):
        assert 1536 < m <= 2048
        acts, ins = inputs(m, COUNT, norms)
        assert all(tile == 0 for _, _, tile in acts)
        kwargs = dict(num_heads=16, num_kv_heads=1, num_index_heads=1, eps=1e-6)

        def project(j, config):
            a, asc, _ = acts[j]
            w, ws = copies[j % len(copies)]
            return g.mxfp8_gemm(a, asc, w, ws, torch.bfloat16,
                                activation_scale_preshuffled=True, config=config)

        # Eager outputs, not capture outputs, establish whether the two sums
        # write the same BF16 QKV and hence the same downstream insert bytes.
        same = torch.equal(project(0, OLD), project(0, None))
        result[m] = {"same_bf16": same}
        for unit in args.units.split(","):
            assert unit in ("gemm", "chain")

            def call(j, config):
                y = project(j, config)
                if unit == "chain":
                    norm_rope_nvfp4_insert(y, *ins[j], **kwargs)
                return y

            calls = {"old": lambda j: call(j, OLD), "new": lambda j: call(j, None)}
            graphs = {arm: capture(fn) for arm, fn in calls.items()}
            samples = {kind: {arm: [] for arm in calls} for kind in ("graph", "eager")}
            for _ in range(3):
                for _ in range(5):
                    for arm, fn in calls.items():
                        samples["graph"][arm].append(measure([graphs[arm]], COUNT))
                    for arm, fn in calls.items():
                        samples["eager"][arm].append(eager(fn))
            summary = {}
            for kind, rounds in samples.items():
                summary[kind] = {
                    "median": {arm: statistics.median(v) for arm, v in rounds.items()},
                    "range": {arm: [min(v), max(v)] for arm, v in rounds.items()},
                    "every_round_faster": all(a < b for a, b in zip(rounds["new"], rounds["old"])),
                    "apart": max(rounds["new"]) < min(rounds["old"]),
                    "rounds": rounds,
                }
            result[m][unit] = summary
            print(m, unit, json.dumps(summary), flush=True)
            del graphs
        del acts, ins
        torch.cuda.empty_cache()
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
