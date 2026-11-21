# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Phase timeline of one fused call (``AKR_EP_TRACE=1``): every workgroup stamps the 100 MHz constant clock when it starts,
leaves the dispatch, sees every rank's rows, has the block table, drains the work queue, sees every rank's finish word and
ends. ``AKR_EP_TRACE=1 python3 benchmarks/moe/fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8/benchmark_timeline.py --world 8
--tokens 64`` replays graphs of distinct inputs (the steady state of a served decode) and prints, per rank, min / median /
max over its workgroups in us after the rank's first workgroup start, and the mean time per workgroup spent in GEMM1 and
GEMM2 tiles. The clocks of two GPUs are not comparable, so a rank's wait for the others shows in ``arrived``."""
import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

EPR, H, I = 48, 5120, 2304
PHASES = ("start", "dispatched", "arrived", "plan", "queue_empty", "fin_seen", "end")


def routes(n, world, seed, dev, kind):
    g = torch.Generator(device=dev).manual_seed(seed)
    s = torch.randn(n, EPR * world, device=dev, generator=g)
    if kind == "zipf":
        perm = torch.randperm(EPR * world, device=dev, generator=torch.Generator(device=dev).manual_seed(4242))
        s[:, perm] += -1.2 * torch.log(torch.arange(1, EPR * world + 1, device=dev).float())
    elif kind == "hot":
        s[:, EPR:] -= 1e4
    v, ids = torch.topk(s, 6, dim=-1)
    return ids.int(), torch.softmax(v, -1)


def worker(rank, world, port, args):
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8 import ExpertParallel, ExpertWeights
    from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8.host import config_for, sized_tokens

    g = torch.Generator(device=dev).manual_seed(100 + rank)
    w = ExpertWeights.from_row_major(
        torch.randint(0, 256, (EPR, 2 * I, H // 2), dtype=torch.uint8, device=dev, generator=g),
        torch.randint(118, 124, (EPR, 2 * I, H // 32), dtype=torch.uint8, device=dev, generator=g),
        torch.randint(0, 256, (EPR, H, I // 2), dtype=torch.uint8, device=dev, generator=g),
        torch.randint(118, 124, (EPR, H, I // 32), dtype=torch.uint8, device=dev, generator=g))
    ep = ExpertParallel(rank, world, dist.group.WORLD, dev)
    n = args.tokens
    ins = []
    for i in range(8):
        ids, tw = routes(n, world, 5 * i + 1000 * rank, dev, args.route)
        ins.append((torch.randn((n, H), device=dev, generator=g).bfloat16(), tw, ids))
    ep(ins[0][0], None, w, ins[0][1], ins[0][2], rows_hint=args.hint or None)
    torch.cuda.synchronize()
    dist.barrier()
    graphs = []
    for x, tw, ids in ins:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            ep(x, None, w, tw, ids, rows_hint=args.hint or None)
        graphs.append(graph)
    torch.cuda.synchronize()
    dist.barrier()
    for _ in range(args.replays):
        for graph in graphs:
            graph.replay()
    torch.cuda.synchronize()
    nw = config_for(sized_tokens(n, args.hint or None))[10]
    stamps = ep.loc[4096:4096 + nw * 32].view(torch.int64).view(nw, 16).cpu()
    t0 = stamps[:, 0].min()
    rel = (stamps[:, :7] - t0).float() * 10 / 1000.0
    out = {"rank": rank, "workgroups": nw}
    for i, name in enumerate(PHASES):
        c = rel[:, i]
        out[name] = [round(float(c.min()), 1), round(float(c.median()), 1), round(float(c.max()), 1)]
    out["gemm1_us_per_workgroup"] = round(float(stamps[:, 8].float().mean()) * 10 / 1000.0, 1)
    out["gemm2_us_per_workgroup"] = round(float(stamps[:, 9].float().mean()) * 10 / 1000.0, 1)
    print(json.dumps(out), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--hint", type=int, default=0)
    parser.add_argument("--route", default="uniform", choices=["uniform", "zipf", "hot"])
    args = parser.parse_args()
    assert os.environ.get("AKR_EP_TRACE") == "1", "set AKR_EP_TRACE=1"
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(worker, args=(args.world, port, args), nprocs=args.world, join=True)


if __name__ == "__main__":
    main()
