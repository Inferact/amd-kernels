# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four-rank GPU time through the library entry, 100 distinct input sets.
Run with four visible GPUs; no vLLM dependency. MAX rank, five rounds of ten
completed 100-call graph replays. Reports kernel time, not serving throughput.
"""
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks4 import SamplerAllGather

def run(rank,port):
    torch.set_num_threads(8);torch.cuda.set_device(rank)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=4,rank=rank)
    dev=torch.device('cuda',rank);comm=SamplerAllGather(rank,dist.group.WORLD,dev);comm.warmup()
    gen=torch.Generator(device=dev).manual_seed(rank+7200)
    for R in (10,16,32,64,128,256):
        counts=[R//4+int(r<R%4) for r in range(4)];n=counts[rank];stride=max(counts)
        src=torch.tensor([i%4*stride+i//4 for i in range(R)],device=dev,dtype=torch.int64)
        cu=torch.arange(0,n*8+1,8,device=dev,dtype=torch.int32)
        sets=[]
        for _ in range(100):
            tok=torch.randint(0,200064,(n,8),generator=gen,device=dev,dtype=torch.int64)
            ns=torch.randint(1,9,(n,),generator=gen,device=dev,dtype=torch.int32)
            nr=8-ns;nan=torch.randint(0,10,(n*8,),generator=gen,device=dev,dtype=torch.int32)
            out=(torch.empty(R,8,device=dev,dtype=torch.int64),*[torch.empty(R,device=dev,dtype=torch.int32) for _ in range(3)])
            sets.append((tok,ns,nr,nan,cu,src,stride,*out))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for args in sets:comm(*args)
        torch.cuda.synchronize();dist.barrier();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for args in sets:comm(*args)
        g.replay();torch.cuda.synchronize();dist.barrier();ts=[]
        for _ in range(5):
            total=0.
            for _ in range(10):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();g.replay();b.record();b.synchronize();total+=a.elapsed_time(b)
            t=torch.tensor([total*1000/(10*len(sets))],dtype=torch.float64);dist.all_reduce(t,op=dist.ReduceOp.MAX);ts.append(t.item())
        if rank==0:print(f'R {R}: {sorted(ts)[2]:.3f} [{min(ts):.3f}-{max(ts):.3f}] us',flush=True)
        del g
    comm.close();dist.destroy_process_group()

if __name__=='__main__':
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    mp.spawn(run,args=(port,),nprocs=4,join=True)
