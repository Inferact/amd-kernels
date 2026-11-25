# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay on one node, measured on the slowest of eight ranks."""
import argparse
import os
import statistics

import torch
import torch.distributed as dist

from amd_kernels.moe.fused_ep_mxfp4_h3584_i3072_e112_top16_ranks8 import ExpertParallel, ExpertWeights
from tests.moe.fused_ep_mxfp4_h3584_i3072_e112_top16_ranks8 import workload


def capture(call, inputs):
    for args in inputs:
        call(*args)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for args in inputs:
            call(*args)
    return graph


def elapsed(graph, calls):
    dist.barrier()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    for _ in range(10):
        graph.replay()
    end.record()
    end.synchronize()
    us = torch.tensor(start.elapsed_time(end) * 1000 / (10 * calls), dtype=torch.float64)
    dist.all_reduce(us, op=dist.ReduceOp.MAX)
    return us.item()


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--tokens", nargs="+", type=int, default=[1, 4, 8, 16, 32, 96, 128, 256])
    args.add_argument("--routes", nargs="+", choices=["uniform", "skew", "hot"], default=["uniform", "skew", "hot"])
    opts = args.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    dist.init_process_group("gloo")
    assert dist.get_world_size() == 8
    w = ExpertWeights.from_row_major(*workload.make_experts(100 + rank, dev))
    ep = ExpertParallel(rank, 8, dist.group.WORLD, dev)
    ep.warm_up(w)

    def call(x, weights, ids):
        return ep(x, None, w, weights, ids)

    for m in opts.tokens:
        for kind in opts.routes:
            inputs = []
            for step in range(4):
                seed = 11 + 193 * rank + 701 * step
                ids, weights = workload.make_routes(m, 8, kind, seed, dev)
                x, _, _ = workload.make_tokens(m, seed + 31, dev)
                inputs.append((x, weights, ids))
            graph = capture(call, inputs)
            us = statistics.median(elapsed(graph, len(inputs)) for _ in range(7))
            if rank == 0:
                print(f"rows={m} routing={kind} us={us:.2f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
