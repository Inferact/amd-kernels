# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MoE slots + shared -> TP4 reduce-scatter -> shard Gemma norm + MXFP8.

Graph replay: 16 distinct inputs, 10 replays, 5 alternated rounds, same ranks.
The baseline uses the graph-safe dense RS (``SlotsReduceScatterComm.reduce_scatter``:
a device counter and a grid-stride staging copy); its slot reduce, shared add and
norm are separate launches. Also times both sides eagerly. Eager arms warm their
allocator after graph execution, before each sample. Eager results include host
dispatch gaps; graph results isolate the fusion with matched graph-safe transport.
Neither is a cross-engine comparison.

Norm baseline: the library's one-GPU ``gemma_add_rmsnorm_mxfp8_row_major``
(row-major codes and AITER scales, as the served SP path). Prints rank0
median/range and max rank.
"""
import json
import statistics
import sys
import numpy as np
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus

H = 6144
CALLS = 16
ROUNDS = 5
REPLAYS = 10
EPS = 1e-6


def _time(fn, n, warm_eager=False):
    dist.barrier()
    torch.cuda.synchronize()
    if warm_eager:
        # Graphs own private pools. Warm the eager allocator again after graph
        # execution, outside timing, on the same >MALL-sized rotating inputs.
        fn()
        torch.cuda.synchronize()
        dist.barrier()
    a,b = torch.cuda.Event(True),torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b)*1000/(n*CALLS)


def _worker(rank, errors, rows_list, output):
    from amd_kernels.collectives.pull_collectives_ranks4 import SlotsReduceScatterComm
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.reduce_slots_shared import reduce_slots_shared
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import reduce_slots_bf16_ref
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
    dev = torch.device('cuda',rank)
    comm = SlotsReduceScatterComm(rank, dist.group.WORLD, dev)
    comm.warm_up()
    w = (torch.randn(H,device=dev)*0.2).bfloat16()
    def norm(x,r):
        return an.gemma_add_rmsnorm_mxfp8_row_major(x,r,w,EPS)
    results = {}
    if rank == 0:
        print(f'GPU {torch.cuda.get_device_properties(dev)}',flush=True)
    for rows in rows_list:
        slots,shared,res = [],[],[]
        # Distinct input seeds; rank-specific slots model TP shards.
        for i in range(CALLS):
            rng = np.random.default_rng(1000*rows+i+rank*100000000)
            def tensor(shape):
                return torch.from_numpy(rng.standard_normal(shape,dtype=np.float32)).to(dev).bfloat16()
            slots.append(tensor((rows,4,H)))
            shared.append(tensor((rows,H)))
            res.append(tensor((rows//4,H)))
        reduced = [reduce_slots_bf16_ref(s) for s in slots]
        def fused():
            for s,h,r in zip(slots,shared,res):
                comm.slots4_reduce_scatter_add_rmsnorm_mxfp8(s,h,r,w,EPS)
        def today():
            for s,h,r in zip(slots,shared,res):
                x = reduce_slots_shared(s, h)
                norm(comm.reduce_scatter(x),r)
        def add_rs_norm():
            for x,h,r in zip(reduced,shared,res):
                norm(comm.reduce_scatter(x+h),r)
        # Compile both baseline norm and the complete chains before capture.
        fused(); today()
        torch.cuda.synchronize()
        graphs = {}
        for k,fn in [('fused',fused),('today_graph',today),('add_rs_norm_graph',add_rs_norm)]:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs[k] = g
        for g in graphs.values():
            g.replay()
            torch.cuda.synchronize()
        times = {k:[] for k in graphs}
        times.update(eager_fused=[],eager_today=[],eager_add_rs_norm=[])
        for round_id in range(ROUNDS):
            keys = list(graphs)
            if round_id%2:
                keys.reverse()
            for k in keys:
                times[k].append(_time(graphs[k].replay,REPLAYS))
            keys = [('eager_fused',fused),('eager_today',today),('eager_add_rs_norm',add_rs_norm)]
            if round_id%2:
                keys.reverse()
            for k,fn in keys:
                times[k].append(_time(fn,2,warm_eager=True))
        every = [None]*4
        dist.all_gather_object(every,times)
        if rank == 0:
            print('rows='+str(rows)+' | '+' | '.join(
                f'{k} {statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}] maxrank '+
                f'{max(statistics.median(e[k]) for e in every):.2f}' for k,v in times.items()),flush=True)
            results[rows] = {'rank0':times,'every':every}
            if output:
                with open(output,'w') as f:
                    json.dump(results,f,indent=1)
        del graphs,slots,shared,res,reduced
        torch.cuda.empty_cache()


def main():
    args = sys.argv[1:]
    output = args.pop(0) if args and args[0].endswith('.json') else None
    rows = list(map(int,args)) or [4096,8192,16384,32768]
    run_on_4_gpus(_worker,rows,output,timeout=3000)


if __name__ == '__main__':
    main()
