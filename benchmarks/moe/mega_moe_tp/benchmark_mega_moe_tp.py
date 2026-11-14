# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MegaMoE TP on 4 ranks: CUDA graph replay of ``--calls`` back-to-back layer calls, each on its own input (the
weights of one rank, 0.9 GB for m3, exceed the LLC), us per call of the slowest rank (median of ``--iters`` replays);
plus GEMM TFLOPS and algorithmic TB/s over the cluster, and rel L2 vs ``reference.py`` of the first input.

  python3 benchmarks/moe/mega_moe_tp/benchmark_mega_moe_tp.py [--shape m3] [--modes ag_rs ar] [--tokens 8 64 ...]
      [--comm fp8 bf16] [--json out.json]
"""

import argparse
import json
import statistics

import torch
import torch.distributed as dist

from tests.multi_gpu import WORLD, run_on_4_gpus


def _graph_us(layer, cases, out, calls, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for c in cases[:3]:
            layer(c.x, c.w, c.ids, out=out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            c = cases[i % len(cases)]
            layer(c.x, c.w, c.ids, out=out)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / calls)
    dist.barrier()
    t = torch.tensor([statistics.median(times)])
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _work(shape, wt, ids_all, tokens):
    """cluster-wide GEMM FLOPs and algorithmic bytes (input, the active experts' MXFP4 weights, routing, output)"""
    H, I, K = shape.model_dim, wt.inter, shape.topk
    flops = 6 * tokens * K * H * I * WORLD
    ids = ids_all.flatten()
    active = int(torch.unique(ids[(ids >= 0) & (ids < shape.experts)]).numel())
    weights = active * (2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32))
    return flops, weights * WORLD + tokens * H * 2 * 2 + tokens * K * 8


def _worker(rank, errors, args):
    from tests.moe.mega_moe_tp.workload import SHAPES, build_weights, make_case, new_layer, rel_l2

    dev = torch.device("cuda", rank)
    shape = SHAPES[args.shape]
    wt = build_weights(shape, rank, WORLD, dev)
    max_local = max(t // WORLD for t in args.tokens)
    rows = []
    for mode in args.modes:
        for comm in args.comm:
            layer = new_layer(shape, wt, rank, WORLD, mode, max_local, comm, "bf16")
            layer.prepare(sorted({-(-t // WORLD) for t in args.tokens}))
            for tokens in args.tokens:
                cases = [make_case(shape, wt, rank, WORLD, mode, tokens, args.route, 1000 + i)
                         for i in range(args.inputs)]
                out = torch.empty_like(cases[0].ref, dtype=torch.bfloat16)
                layer(cases[0].x, cases[0].w, cases[0].ids, out=out)
                torch.cuda.synchronize()
                e = rel_l2(out, cases[0].ref)
                us = _graph_us(layer, cases, out, args.calls, args.iters)
                if layer.poll_errors():
                    errors.append(f"rank {rank} {mode} {comm} M={tokens}: watchdog error")
                flops, nbytes = _work(shape, wt, cases[0].ids_all, tokens)
                cfg = layer.launch_config(tokens // WORLD)
                plan = (f"mt{cfg.mt}{' lb' if cfg.lb else ''}" if cfg is not None
                        else "big-M " + ",".join(map(str, layer.bigm_plan(tokens // WORLD)[0])))
                rows.append({"shape": shape.name, "mode": mode, "comm": comm, "M": tokens, "config": plan,
                             "us": round(us, 1), "TFLOPS": round(flops / us / 1e6, 1),
                             "TB/s": round(nbytes / us / 1e6, 2), "rel_l2": round(e, 4)})
                del cases
            layer.close()
    if rank == 0:
        cols = list(rows[0])
        print("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols), flush=True)
        for r in rows:
            print("| " + " | ".join(str(r[c]) for c in cols) + " |", flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"gpu": torch.cuda.get_device_properties(dev).name, "args": vars(args), "rows": rows}, f,
                          indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", default="m3", choices=["m3", "glm5"])
    p.add_argument("--modes", nargs="*", default=["ag_rs", "ar"], choices=["ag_rs", "ar"])
    p.add_argument("--tokens", type=int, nargs="*", default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
                   help="global tokens per call (multiples of 4)")
    p.add_argument("--comm", nargs="*", default=["fp8"], choices=["fp8", "bf16"])
    p.add_argument("--route", default="random", choices=["balanced", "random", "hot", "subset"])
    p.add_argument("--inputs", type=int, default=4, help="distinct inputs the graph's calls cycle over")
    p.add_argument("--calls", type=int, default=10, help="layer calls per graph")
    p.add_argument("--iters", type=int, default=20, help="timed graph replays")
    p.add_argument("--json")
    args = p.parse_args()
    bad = [t for t in args.tokens if t <= 0 or t % WORLD]
    if bad:
        p.error(f"tokens {bad}: positive multiples of {WORLD}")
    run_on_4_gpus(_worker, args, timeout=7200)


if __name__ == "__main__":
    main()
