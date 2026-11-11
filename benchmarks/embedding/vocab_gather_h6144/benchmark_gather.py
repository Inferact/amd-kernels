# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four-rank embedding unit latency; warm immutable weights, 100 distinct anchor batches.

The full-draft A/B is the cold-serving evidence, since a small embedding-only graph
cannot reproduce the 430 MB of intervening layer weights at R=1.
"""
import argparse
import statistics
import torch
import torch.distributed as dist
from amd_kernels.embedding.vocab_gather_h6144 import VocabGather
from tests.multi_gpu import run_on_4_gpus


def worker(rank,errors,requests):
    device=torch.device('cuda',rank)
    torch.manual_seed(31+rank)
    weight=torch.randn((50016,6144),device=device,dtype=torch.bfloat16)
    lookup=VocabGather(weight,rank,dist.group.WORLD)
    lookup.warm_up()
    for nr in requests:
        inputs=[]
        for i in range(100):
            ids=torch.full((nr,7),200058,device=device,dtype=torch.int64)
            gen=torch.Generator(device=device).manual_seed(1000*nr+i)
            ids[:,0]=torch.randint(0,200064,(nr,),device=device,generator=gen)
            inputs.append(ids.flatten())
        graph=torch.cuda.CUDAGraph();outputs=[]
        with torch.cuda.graph(graph):
            for ids in inputs:outputs.append(lookup(ids))
        for _ in range(3):graph.replay()
        torch.cuda.synchronize()
        times=[]
        for _ in range(5):
            dist.barrier();torch.cuda.synchronize()
            start,end=torch.cuda.Event(True),torch.cuda.Event(True)
            start.record()
            for _ in range(10):graph.replay()
            end.record();end.synchronize()
            times.append(start.elapsed_time(end))
        every=[None]*4;dist.all_gather_object(every,times)
        maximum=[max(values[i] for values in every) for i in range(5)]
        if rank==0:print(nr,statistics.median(maximum),[min(maximum),max(maximum)],'us/call',flush=True)
    lookup.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reqs',type=int,nargs='+',default=[1,10,16,32,64,128,256]);a=p.parse_args()
    run_on_4_gpus(worker,a.reqs,timeout=300)
