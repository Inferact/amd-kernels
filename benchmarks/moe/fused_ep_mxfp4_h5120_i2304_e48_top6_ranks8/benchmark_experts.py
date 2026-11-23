# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused expert-parallel MoE, graph replay of distinct calls on 2 / 4 / 8 GPUs (one process per GPU, ``torchrun``).

``torchrun --nproc-per-node 8 benchmarks/moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8/benchmark_experts.py``.
``--ninputs`` calls on distinct tokens and routes, one CUDA graph each (``--chain`` calls per graph), replayed back to
back; us per call = max over the ranks; the rounds run A B ... B A over the route kinds (uniform, Zipf 1.2, all routes
on rank 0, every route on the token's own rank) and then over per-rank token counts (``--imb``: one rank with T, the
others T / 8; or T (r + 1) / 8). The weights (0.9 GB per rank) are shared by all calls and exceed the last-level cache.
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

EPR, TOPK, H, I = 48, 6, 5120, 2304


def make_weights(rank, dev):
    from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 import ExpertWeights

    g = torch.Generator(device=dev).manual_seed(9000 + rank)

    def rnd(lo, hi, shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device=dev, generator=g)

    return ExpertWeights.from_row_major(rnd(0, 256, (EPR, 2 * I, H // 2)), rnd(120, 125, (EPR, 2 * I, H // 32)),
                                        rnd(0, 256, (EPR, H, I // 2)), rnd(120, 125, (EPR, H, I // 32)))


def make_inputs(n_tokens, route, rank, world, dev, n, seed):
    """``n`` distinct (tokens, route weights, ids): ``uniform`` | ``zipf`` | ``hot`` | ``local``."""
    experts = EPR * world
    xs, ws, ids = [], [], []
    for i in range(n):
        g = torch.Generator(device=dev).manual_seed(seed + 1000 * rank + i)
        x = torch.randn(n_tokens, H, dtype=torch.bfloat16, device=dev, generator=g)
        s = torch.randn(n_tokens, experts, dtype=torch.float32, device=dev, generator=g)
        if route == "zipf":
            perm = torch.randperm(experts, device=dev, generator=torch.Generator(device=dev).manual_seed(4242))
            logp = torch.empty(experts, device=dev)
            logp[perm] = -1.2 * torch.log(torch.arange(1, experts + 1, device=dev).float())
            u = torch.rand(n_tokens, experts, device=dev, generator=g).clamp_(1e-9, 1 - 1e-9)
            s = logp - torch.log(-torch.log(u))
        elif route == "hot":
            s[:, EPR:] -= 1e4
        elif route == "local":
            s[:, : rank * EPR] -= 1e4
            s[:, (rank + 1) * EPR:] -= 1e4
        v, idx = torch.topk(s, TOPK, dim=-1)
        xs.append(x)
        ws.append(torch.softmax(v, -1).contiguous())
        ids.append(idx.to(torch.int32).contiguous())
    return xs, ws, ids


def barrier():
    torch.cuda.synchronize()
    dist.barrier()


def timed(graphs, reps, dev, world, calls_per_graph):
    barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        for g in graphs:
            g.replay()
    end.record()
    torch.cuda.synchronize()
    us = torch.tensor(start.elapsed_time(end) * 1000.0 / (reps * len(graphs) * calls_per_graph), dtype=torch.float64,
                      device=dev)
    per_rank = [torch.zeros_like(us) for _ in range(world)]
    dist.all_gather(per_rank, us)
    per_rank = [x.item() for x in per_rank]
    return max(per_rank), sum(per_rank) / world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,16,32,64,128,192,384,512,768")
    parser.add_argument("--routes", default="uniform,zipf,hot")
    parser.add_argument("--ninputs", type=int, default=100)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--chain", type=int, default=1)
    parser.add_argument("--imb", default="")
    parser.add_argument("--imb-tokens", default="192,384")
    parser.add_argument("--mxfp8", action="store_true", help="MXFP8 rows instead of bf16")
    parser.add_argument("--hint", type=float, default=0, help="rows_hint = hint x the rank's token count")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("gloo")
    from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 import ExpertParallel
    from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8.reference import quantize_mxfp8

    w = make_weights(rank, dev)
    ep = ExpertParallel(rank, world, dist.group.WORLD, dev)
    scenarios = [(str(t), [t] * world) for t in map(int, args.tokens.split(",")) if t]
    for t in [int(v) for v in args.imb_tokens.split(",") if v]:
        for kind in [k for k in args.imb.split(",") if k]:
            counts = [t] + [max(1, t // 8)] * (world - 1) if kind == "onehot" else [
                max(1, t * (r + 1) // world) for r in range(world)]
            scenarios.append((f"imb_{kind}_{t}", counts))
    results = []
    for label, counts in scenarios:
        routes = args.routes.split(",") if not label.startswith("imb_") else ["uniform"]
        graphs = {}
        for route in routes:
            xs, ws, ids = make_inputs(counts[rank], route, rank, world, dev, args.ninputs, 5000)
            hint = int(args.hint * counts[rank]) or None
            if args.mxfp8:
                q = [quantize_mxfp8(x) for x in xs]

                def call(i, q=q, ws=ws, ids=ids, hint=hint):
                    return ep(q[i][0], q[i][1], w, ws[i], ids[i], rows_hint=hint)
            else:

                def call(i, xs=xs, ws=ws, ids=ids, hint=hint):
                    return ep(xs[i], None, w, ws[i], ids[i], rows_hint=hint)
            barrier()
            call(0)
            barrier()
            gs = []
            for first in range(0, args.ninputs, args.chain):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=torch.cuda.Stream()):
                    for i in range(first, min(first + args.chain, args.ninputs)):
                        call(i)
                gs.append(g)
            barrier()
            for g in gs[:10]:
                g.replay()
            barrier()
            graphs[route] = (gs, (xs, ws, ids))
        samples = {r: [] for r in routes}
        for r in routes + routes[::-1]:
            samples[r].append(timed(graphs[r][0], args.reps, dev, world, args.chain))
        for r in routes:
            record = {"tokens": label, "counts": counts, "route": r, "us_max": [x[0] for x in samples[r]],
                      "us_mean": [x[1] for x in samples[r]]}
            if rank == 0:
                results.append(record)
                print(json.dumps(record), flush=True)
        del graphs
        barrier()
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
