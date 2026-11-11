# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import os,socket
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks4.reference import reference


def fixtures(counts,seed,has=True,ncols=8,fixed_cu=None):
    g=torch.Generator().manual_seed(seed)
    local=[]
    for r,n in enumerate(counts):
        tok=torch.randint(-(1<<60),1<<60,(n,ncols),generator=g,dtype=torch.int64)
        ns=torch.randint(1,ncols+1,(n,),generator=g,dtype=torch.int32)
        nr=torch.randint(0,8,(n,),generator=g,dtype=torch.int32)
        lens=torch.randint(1,9,(n,),generator=g,dtype=torch.int32)
        cu=torch.cat([torch.zeros(1,dtype=torch.int32),lens.cumsum(0).int()])
        if fixed_cu is not None:cu=fixed_cu[r]
        nan=torch.randint(0,200065,(int(cu[-1]),),generator=g,dtype=torch.int32) if has else None
        local.append((tok,ns,nr,nan,cu))
    return local


def run(rank,port):
    torch.set_num_threads(8);torch.cuda.set_device(rank)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=4,rank=rank)
    dev=torch.device('cuda',rank)
    from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks4 import SamplerAllGather
    comm=SamplerAllGather(rank,dist.group.WORLD,dev);comm.warmup()
    # All variants share a single launcher; shape, NaN presence and columns are runtime values.
    import flydsl.compiler.jit_function as jf
    from unittest.mock import patch
    with patch.object(jf.MlirCompiler,'compile',side_effect=AssertionError('compile after warmup')):
        for ic,counts in enumerate(([1,0,0,0],[3,3,2,2],[4]*4,[8]*4,[16]*4,[32]*4,[64]*4,[0,0,256,0],[5,0,8,0],[0]*4)):
            stride=max(max(counts),1);R=sum(counts)
            src=torch.tensor([r*stride+i for r,n in enumerate(counts) for i in range(n)],dtype=torch.int64)
            if R:src=src[torch.randperm(R,generator=torch.Generator().manual_seed(ic))]
            for has in (False,True):
                for ncols in (1,7,8):
                    local=fixtures(counts,100+ic,has,ncols)
                    inp=[a.to(dev) if a is not None else None for a in local[rank]]
                    # Counts/prefixes may be views with only natural four-byte alignment.
                    for k in (1,2,3,4):
                        if inp[k] is not None:
                            buf=torch.empty(inp[k].numel()+1,device=dev,dtype=torch.int32)
                            buf[1:].copy_(inp[k]);inp[k]=buf[1:]
                    if ncols==7:
                        strided=torch.empty(counts[rank],12,device=dev,dtype=torch.int64)[:,:7]
                        strided.copy_(inp[0]);inp[0]=strided
                    smap=src.to(dev)
                    out=(torch.empty(R,8,device=dev,dtype=torch.int64),torch.empty(R+1,device=dev,dtype=torch.int32)[1:],
                         torch.empty(R+1,device=dev,dtype=torch.int32)[1:],torch.empty(R+1,device=dev,dtype=torch.int32)[1:] if has else None)
                    def call():return comm(*inp,smap,stride,*out)
                    call();torch.cuda.synchronize();want=reference(local,src,stride)
                    for a,b in zip(out,want):
                        if a is not None:assert torch.equal(a.cpu(),b),(rank,counts,has,ncols)
                    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):call()
                    torch.cuda.synchronize();dist.barrier()
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(10):call()
                    for rep in range(3):
                        # Same data/map pointers, fresh payload and reordered output every replay.
                        new=fixtures(counts,1000+rep+ic,has,ncols,[a[4] for a in local])
                        for dst,ss in zip(inp,new[rank]):
                            if dst is not None:dst.copy_(ss)
                        now=src.flip(0) if rep%2 else src
                        smap.copy_(now.to(dev))
                        for a in out:
                            if a is not None:a.fill_(-123)
                        dist.barrier();graph.replay();torch.cuda.synchronize()
                        want=reference(new,now,stride)
                        for a,b in zip(out,want):
                            if a is not None:assert torch.equal(a.cpu(),b),(rank,ic,has,ncols,rep)
                    del graph
    # Thousands of double-slot uses, sequence wrap, duplicate map entries.
    counts=[64]*4;local=fixtures(counts,9);inp=[a.to(dev) for a in local[rank]]
    src=torch.tensor([0,63,64,127,128,191,192,255]*32,dtype=torch.int64);smap=src.to(dev)
    out=(torch.empty(256,8,device=dev,dtype=torch.int64),*[torch.empty(256,device=dev,dtype=torch.int32) for _ in range(3)])
    comm.counter.fill_(2147483645)
    # All ranks' uncached flags must be synchronized to the reset sequence first.
    from amd_kernels.common.hip_runtime import load_hip_runtime
    import ctypes
    lib=load_hip_runtime();lib.hipMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
    v=ctypes.c_int32(2147483645)
    assert lib.hipMemcpy(ctypes.c_void_p(comm.flags[rank]),ctypes.byref(v),4,1)==0
    dist.barrier()
    for _ in range(1000):comm(*inp,smap,64,*out)
    torch.cuda.synchronize();want=reference(local,src,64)
    for a,b in zip(out,want):assert torch.equal(a.cpu(),b)
    comm.close();dist.destroy_process_group()


def test_four_rank_graph_bytes_and_wrap():
    if torch.cuda.device_count()<4:pytest.skip('requires four MI355X GPUs')
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    mp.spawn(run,args=(port,),nprocs=4,join=True)
