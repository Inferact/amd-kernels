# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The full shared expert (I 3072) on R own rows (``shared_expert_full``, sequence-parallel prefill) vs the TP2
shard (I 1536) on all 2 R rows (``shared_dense`` on a shard copy); ``--sweep``: every gate/up schedule and tile of
the full expert. Graph replay of ``--copies`` inputs, median of ``--rounds``, us per call."""
import argparse
import json
import statistics

import torch

from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import SharedWeights
from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.shared_dense import full_config_for, shared_dense, shared_dense_full
from tests.moe.mxfp4_h6144_i1536_e128_top4.workload import make_full_shared, make_mxfp8


def timed(fn, inputs, rounds, replays):
    fn(*inputs[0])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x in inputs:
            fn(*x)
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (replays * len(inputs)))
    return statistics.median(samples)


def configs(r):
    out = []
    for t1, sched in (((4, 4), "deferred"), ((2, 4), "deferred"), ((4, 4), "early"), ((4, 4), "single"),
                      ((2, 4), "single")):
        if sched == "single" and ((r + 255) // 256) * 24 * (4 // t1[0]) > 256:
            continue
        for t2 in ((4, 4), (2, 4)):
            for nt in (False, True):
                out.append(((t1, sched), (t2, nt)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384])
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--copies", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--replays", type=int, default=10)
    args = parser.parse_args()
    raw, shards = make_full_shared()
    full = SharedWeights.from_row_major(*raw, tp_rank=0)
    shard = SharedWeights.from_row_major(*shards[0])
    for r in args.rows:
        own = [make_mxfp8(r, seed=r + i) for i in range(args.copies)]
        every = [make_mxfp8(2 * r, seed=r + i) for i in range(args.copies)]
        rec = {"rows": r, "shard_on_2r_us": timed(lambda q, s: shared_dense(q, s, shard), every, args.rounds,
                                                  args.replays)}
        rec["full_us"] = timed(lambda q, s: shared_dense_full(q, s, full), own, args.rounds, args.replays)
        rec["full_config"] = str(full_config_for(r))
        if args.sweep:
            rec["sweep"] = {str(c): timed(lambda q, s, c=c: shared_dense_full(q, s, full, config=c), own,
                                          args.rounds, args.replays) for c in configs(r)}
        rec["full_vs_shard"] = f"{rec['full_us'] / rec['shard_on_2r_us'] - 1:+.1%}"
        print(json.dumps(rec), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
