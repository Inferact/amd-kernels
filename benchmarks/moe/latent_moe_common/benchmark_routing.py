# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the entries add to a call: graph replay and eager host time per call, entry against the direct shape-package
call, at served rows. One GPU: router (with the sort) and experts; on 8 ranks (torchrun): latent tail and layer tail.
Both calls take the same arguments. Graph: 20 calls per graph, min of 15 rounds of 5 replays, us per call; host: min
over 50 batches of 100 calls of the time the Python call takes (no synchronize inside the batch), us per call; 8
ranks: the max over ranks.
Run: python3 benchmarks/moe/latent_moe_common/benchmark_routing.py
     torchrun --nproc_per_node 8 benchmarks/moe/latent_moe_common/benchmark_routing.py
"""
import importlib
import os
import time

import torch
import torch.distributed as dist

from amd_kernels.moe import latent_moe_common as E
from amd_kernels.moe.latent_moe_common import experts, latent_tail, layer_tail, router


def graph_us(fns, calls=20, rounds=15):
    """Per function, min over ``rounds`` of 5 replays of a graph of ``calls`` calls; the graphs alternate."""
    graphs = []
    for fn in (fns[0], *fns):  # the first graph captured replays ~0.3 us slower whichever call it holds: discarded
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
        graph.replay()
        graphs.append(graph)
    graphs = graphs[1:]
    torch.cuda.synchronize()
    best = [float("inf")] * len(fns)
    for _ in range(rounds):
        for i, graph in enumerate(graphs):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                graph.replay()
            b.record()
            b.synchronize()
            best[i] = min(best[i], a.elapsed_time(b) * 1e3 / (5 * calls))
    return best


def host_us(fns, calls=100, rounds=50):
    """Per function, min over ``rounds`` batches of ``calls`` calls of the host time a call takes; batches alternate."""
    best = [float("inf")] * len(fns)
    for _ in range(rounds):
        for i, fn in enumerate(fns):
            if dist.is_initialized():
                dist.barrier()
            t = time.perf_counter()
            for _ in range(calls):
                fn()
            best[i] = min(best[i], (time.perf_counter() - t) / calls * 1e6)
            torch.cuda.synchronize()
    return best


def worst(v):
    if not dist.is_initialized():
        return v
    t = torch.tensor([v], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


def row(name, m, direct, entry):
    g = [worst(v) for v in graph_us((direct, entry))]
    h = [worst(v) for v in host_us((direct, entry))]
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"| {name} | {m} | {g[0]:.2f} | {g[1]:.2f} | {h[0]:.2f} | {h[1]:.2f} | {h[1] - h[0]:+.2f} |", flush=True)


def single_gpu():
    from tests.moe.mxfp4_h3584_i3072_e896_top16.workload import make_checkpoint

    dev = torch.device("cuda")
    rh = importlib.import_module(router._SHAPE_HOSTS[(896, 7168, 16)])
    eh = importlib.import_module(experts._SHAPE_HOSTS[(896, 768, 1792)])
    w = (torch.randn(896, 7168, device=dev) * 0.02).to(torch.bfloat16)
    bias = torch.randn(896, device=dev) * 0.1
    ck = eh.MoeWeights.from_checkpoint(*make_checkpoint())
    weights = E.moe_weights(ck.w13, ck.s13, ck.w2, ck.s2)
    for m in (1, 16, 192):
        x = torch.randn(m, 7168, device=dev).to(torch.bfloat16)
        out = (torch.empty(m, 16, device=dev), torch.empty(m, 16, dtype=torch.int32, device=dev))
        aux = E.router_aux(w, m, dev, zero_moe_buf=False)
        row("router + sort", m,
            lambda: rh.router_sigmoid_bias_top16_e896_h7168(x, w, bias, 1.0, out, shuffled=False, aux=aux),
            lambda: E.router_sigmoid_bias_topk(x, w, bias, 1.0, out, aux=aux))
        sort = E.moe_sort_from_aux(aux, weights=weights)
        codes, scales = E.quantize_moe_activations(torch.randn(m, 3584, device=dev).to(torch.bfloat16), weights=weights)
        y = torch.empty(m, 3584, dtype=torch.bfloat16, device=dev)
        row("experts", m, lambda: eh.mxfp4_moe_experts(codes, scales, weights, sort, y, fused=True, reduce=True),
            lambda: E.mxfp4_moe_experts(codes, scales, weights, sort, y))


def eight_ranks():
    from tests.moe.layer_tail_k2_h7168_ranks8.inputs import layer_inputs, moe_weights

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    dev, group = torch.device("cuda", rank), dist.group.WORLD
    lh = importlib.import_module(latent_tail._SHAPE_HOSTS[(896, 3584, 8)])
    comm = E.latent_tail_comm(rank, group, dev, up_rows=896, latent=3584)
    for m in (1, 64, 256):
        x = torch.randn(m, 3584, device=dev).to(torch.bfloat16)
        nw = torch.ones(3584, dtype=torch.bfloat16, device=dev)
        up = (torch.randn(896, 3584, device=dev) * 0.02).to(torch.bfloat16)
        shared = torch.zeros(m, 896, dtype=torch.bfloat16, device=dev)
        row("latent tail", m, lambda: lh.allreduce_latent_rmsnorm_up_proj(comm, x, nw, up, shared),
            lambda: E.allreduce_latent_rmsnorm_up_proj(comm, x, nw, up, shared))
    comm.close()
    kh = importlib.import_module(layer_tail._SHAPE_HOSTS[(7168, 1536, 896, 8)])
    runner = E.layer_tail_runner(dev, group, rank, hidden=7168, o_proj_k=1536, experts=896)
    _, wd = moe_weights(rank, dev)
    for s in (4, 16):
        w = layer_inputs(rank, dev, s)
        kw = dict(core=w["core"], w_o=w["w_o"], prefix=w["prefix"], blocks=w["blocks"], ares_nw=w["ares_nw"],
                  ares_qk=w["ares_qk"], in_nw=w["in_nw"], **wd, out=torch.empty(s, 7168, dtype=torch.bfloat16,
                                                                                device=dev), layer=131)
        key = kh.K2Build(tokens=s, nblocks=4)
        build = {"tokens": s, "nblocks": 4, "m1b_qlen": 0, "dsa": False}

        def direct():
            runner.step_begin()
            kh.k2_launch(key, **kw, **runner.launch_args(), **runner.peer_args())

        def entry():
            runner.step_begin()
            E.layer_tail_launch(runner, build, **kw)

        row("layer tail", s, direct, entry)
    torch.cuda.synchronize()
    runner.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.environ.get("WORLD_SIZE"):
        eight_ranks()
    else:
        print("| op | M | graph direct | graph entry | host direct | host entry | host added |")
        print("|---|---|---|---|---|---|---|")
        single_gpu()
