# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the bf16 GEMM entries add over a direct call of the routed package: CUDA-graph replay (the graph holds the
same kernels), the host time of an eager call (enqueue only), of the entry's routing (dispatch) and of the call's
reason; min over the rounds of the per-round mean, us per call. Eight-rank ops: the max over the ranks.

Run: python3 benchmarks/gemm/bf16_common/benchmark_routing.py --op latent_down   (1 GPU)
     python3 benchmarks/gemm/bf16_common/benchmark_routing.py --op linear_gather (8 GPUs)
"""
import argparse
import importlib
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N_GRAPH, REPS, N_EAGER = 24, 10, 200


def _sync():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _worst(v):
    if not dist.is_initialized():
        return v
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, v)
    return [max(e[i] for e in every) for i in range(len(v))]


def timeit(fns, rounds):
    """[(graph us, eager us)] per fn."""
    graphs = []
    for fn in fns:
        fn()
        _sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(N_GRAPH):
                fn()
        graphs.append(g)
    gt, et = [[] for _ in fns], [[] for _ in fns]
    for r in range(rounds):
        for i in range(len(fns)) if r % 2 == 0 else range(len(fns) - 1, -1, -1):
            _sync()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(REPS):
                graphs[i].replay()
            b.record()
            b.synchronize()
            gt[i].append(a.elapsed_time(b) * 1e3 / (REPS * N_GRAPH))
            _sync()
            t = time.perf_counter()
            for _ in range(N_EAGER):
                fns[i]()
            et[i].append((time.perf_counter() - t) * 1e6 / N_EAGER)
            torch.cuda.synchronize()
    return [(min(_worst(g)), min(_worst(e))) for g, e in zip(gt, et)]


def host_us(fn, rounds):
    """Host us per call of a host-only function (the reason)."""
    ts = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(N_EAGER):
            fn()
        ts.append((time.perf_counter() - t) * 1e6 / N_EAGER)
    return min(_worst([min(ts)]))


def _print(rows, res, dispatch, reason):
    (ge, ee), (gd, ed) = res
    print(f"| {rows} | {gd:.2f} | {ge:.2f} | {ed:.2f} | {ee:.2f} | {dispatch:.2f} | {reason:.2f} |", flush=True)


HEADER = ("| rows | graph direct | graph entry | host direct | host entry | dispatch | reason |\n"
          "|---|---|---|---|---|---|---|")


def latent_down(rows_list, rounds):
    from amd_kernels.gemm.bf16_common import (bf16_latent_down_mxfp4_quant, bf16_latent_down_mxfp4_quant_blocker,
                                              bf16_latent_down_mxfp4_quant_shapes)
    from amd_kernels.gemm.bf16_common import latent_down as entry

    for key in sorted(bf16_latent_down_mxfp4_quant_shapes()):
        n, k = key
        name = entry._ROUTES.hosts[key]
        direct = getattr(importlib.import_module(name), name.split(".")[-2])
        w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
        print(f"latent down {key}\n{HEADER}")
        for rows in rows_list:
            x = torch.randn(rows, k, device="cuda").bfloat16()
            out = (torch.empty(rows, n // 2, dtype=torch.uint8, device="cuda"),
                   torch.empty(rows, n // 32, dtype=torch.uint8, device="cuda"))
            _print(rows, timeit([lambda: bf16_latent_down_mxfp4_quant(x, w, out), lambda: direct(x, w, out)], rounds),
                   host_us(lambda: entry._op(entry._ROUTES.host(tuple(w.shape))), rounds),
                   host_us(lambda: bf16_latent_down_mxfp4_quant_blocker(x, w, out), rounds))


def _linear_gather_rank(rank, port, rows_list, rounds):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=8)
    from amd_kernels.gemm.bf16_common import (bf16_linear_gather, bf16_linear_gather_blocker,
                                              bf16_linear_gather_shapes, linear_gather,
                                              linear_gather_comm, linear_gather_pack_weight,
                                              linear_gather_rows_buffer)

    dev = torch.device("cuda", rank)
    for key in sorted(bf16_linear_gather_shapes()):
        n_shard, k, world = key
        if world != 8:
            continue
        host = importlib.import_module(linear_gather._ROUTES.hosts[key])
        comm = linear_gather_comm(rank, dist.group.WORLD, dev, n_shard=n_shard, k=k)
        wp = linear_gather_pack_weight((torch.randn(n_shard, k, device=dev) * 0.01).bfloat16())
        nw = torch.ones(n_shard * world, dtype=torch.bfloat16, device=dev)
        if rank == 0:
            print(f"linear gather {key}\n{HEADER}")
        for rows in rows_list:
            xp = linear_gather_rows_buffer(rows, dev, w_packed=wp).normal_()
            out = torch.empty(rows, n_shard * world, dtype=torch.bfloat16, device=dev)
            res = timeit([lambda: bf16_linear_gather(comm, xp, wp, nw, rows, 1e-5, out),
                          lambda: host.context_proj_gather_norm(comm, xp, wp, nw, rows, 1e-5, out)], rounds)
            dispatch = host_us(lambda: linear_gather._ROUTES.host(linear_gather._comm_key(type(comm))), rounds)
            reason = host_us(lambda: bf16_linear_gather_blocker(comm, xp, wp, nw, rows, 1e-5, out), rounds)
            if rank == 0:
                _print(rows, res, dispatch, reason)
        comm.close()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["latent_down", "linear_gather"], default="latent_down")
    p.add_argument("--rows", type=int, nargs="+", default=[1, 64, 192])
    p.add_argument("--rounds", type=int, default=15)
    a = p.parse_args()
    if a.op == "latent_down":
        latent_down(a.rows, a.rounds)
    else:
        mp.spawn(_linear_gather_rank, args=(29500 + os.getpid() % 300, a.rows, a.rounds), nprocs=8)


if __name__ == "__main__":
    main()
