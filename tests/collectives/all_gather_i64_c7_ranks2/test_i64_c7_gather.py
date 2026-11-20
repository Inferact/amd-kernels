# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import socket
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from amd_kernels.collectives.all_gather_i64_c7_ranks2.reference import reference

def run(rank,port):
    torch.set_num_threads(8);torch.cuda.set_device(rank)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',world_size=2,rank=rank)
    dev=torch.device('cuda',rank)
    from amd_kernels.collectives.all_gather_i64_c7_ranks2 import I64C7AllGather
    comm=I64C7AllGather(rank,dist.group.WORLD,dev);comm.warmup()
    from unittest.mock import patch
    import flydsl.compiler.jit_function as jf
    with patch.object(jf.MlirCompiler,'compile',side_effect=AssertionError('compile after warmup')):
        for case,counts in enumerate(([1,0],[3,2],[4]*2,[8]*2,[16]*2,[32]*2,[64]*2,[128]*2,[0,256],[256,0],[5,0],[0,8],[0]*2)):
            stride=max(max(counts),1);R=sum(counts)
            src=torch.tensor([r*stride+i for r,n in enumerate(counts) for i in range(n)],dtype=torch.int64)
            src=src[torch.randperm(R,generator=torch.Generator().manual_seed(case))]
            # A larger source with stride14 exercises non-contiguous row views and source padding.
            x=torch.empty(counts[rank],14,device=dev,dtype=torch.int64)[:,:7]
            out=torch.empty(R,7,device=dev,dtype=torch.int64);smap=src.to(dev)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):comm(x,smap,stride,out)
            torch.cuda.synchronize();dist.barrier();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):comm(x,smap,stride,out)
            for rep in range(5):
                gen=torch.Generator().manual_seed(1000+case*10+rep)
                local=[torch.randint(-(1<<60),1<<60,(n,7),generator=gen,dtype=torch.int64) for n in counts]
                x.copy_(local[rank].to(dev));now=src.flip(0) if rep%2 else src
                smap.copy_(now.to(dev));out.fill_(-123);dist.barrier();g.replay();torch.cuda.synchronize()
                assert torch.equal(out.cpu(),reference(local,now,stride)),(rank,case,rep)
            del g
    comm.close();dist.destroy_process_group()

def test_two_rank_graph_bytes_and_padding():
    if torch.cuda.device_count()<2:pytest.skip('requires two MI355X GPUs')
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    mp.spawn(run,args=(port,),nprocs=2,join=True)
