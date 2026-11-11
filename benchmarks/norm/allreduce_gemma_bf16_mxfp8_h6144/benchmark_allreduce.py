# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One row count per four-rank process set; 100 distinct inputs, alternated MAX-rank graph rounds. Baseline: AITER's
custom all-reduce, the one-GPU bf16 norm, then (MXFP8) the FlyDSL quantizer of its output (row-major codes)."""
import argparse
import json
import statistics
import sys
import torch
import torch.distributed as dist
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144 import Bf16ArNormComm
from amd_kernels.gemm.mxfp8_common.gemm import mxfp8_e4m3_quantize_flydsl
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an


def worker(rank, errors, m, tile, rounds):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
    torch.manual_seed(12000+rank)
    dev=torch.device('cuda',rank)
    ca=CustomAllreduce(group=dist.group.WORLD,device=dev,max_size=64*1024*1024)
    comm=Bf16ArNormComm(rank,dist.group.WORLD,dev)
    w=torch.randn(6144,device=dev,dtype=torch.bfloat16)*0.2
    xs=[torch.randn(m,6144,device=dev,dtype=torch.bfloat16) for _ in range(100)]
    rs=[torch.randn_like(x) for x in xs]
    def old(i):
        a=ca.custom_all_reduce(xs[i])
        y,res=an.gemma_add_rmsnorm(a,rs[i],w,1e-6)
        if tile<0:return y,res
        return (*mxfp8_e4m3_quantize_flydsl(y),res)
    def new(i):
        if tile<0:return comm.ar_add_rmsnorm(xs[i],rs[i],w)
        return comm.ar_add_rmsnorm_mxfp8(xs[i],rs[i],w,1e-6,tile)
    for fn in (old,new):fn(0)
    torch.cuda.synchronize()
    graphs=[]
    for cm,fn in ((ca,old),(comm,new)):
        g=torch.cuda.CUDAGraph()
        with cm.capture(),torch.cuda.graph(g):
            for i in range(100):fn(i)
        graphs.append(g)
    for g in graphs:g.replay()
    torch.cuda.synchronize()
    ts=[[],[]]
    for r in range(rounds):
        for k in ((0,1) if r%2==0 else (1,0)):
            dist.barrier();torch.cuda.synchronize()
            a,b=torch.cuda.Event(True),torch.cuda.Event(True)
            a.record()
            for _ in range(10):graphs[k].replay()
            b.record();b.synchronize()
            ts[k].append(a.elapsed_time(b)*1e3/1000)
    every=[None]*4
    dist.all_gather_object(every,ts)
    if rank==0:
        vals=[[max(v[k][r] for v in every) for r in range(rounds)] for k in range(2)]
        print(json.dumps(dict(rows=m,tile=tile,old=vals[0],new=vals[1],medians=[statistics.median(v) for v in vals])),flush=True)
    comm.close()
    ca.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--rows',nargs='+',type=int,default=[7,70,112,224,448,896,1792]);p.add_argument('--tile',type=int,default=16);p.add_argument('--rounds',type=int,default=5);a=p.parse_args()
    from tests.multi_gpu import run_on_4_gpus
    for m in a.rows:run_on_4_gpus(worker,m,a.tile,a.rounds)
