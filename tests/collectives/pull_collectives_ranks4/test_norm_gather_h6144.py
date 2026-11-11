# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Chunk-ready dense norm/all-gather against the dense tiled norm's exact bits."""
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus


def _worker(rank,errors,rows_list):
    from amd_kernels.collectives.pull_collectives_ranks4.norm_gather_h6144 import NormAllGatherComm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.host import gemma_add_rmsnorm_mxfp8
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import untile_codes
    dev=torch.device('cuda',rank)
    comm=NormAllGatherComm(rank,dist.group.WORLD,dev)
    comm.warm_up()
    assert comm.supports(0) and not comm.supports(1) and not comm.supports(-32)
    torch.manual_seed(100+rank)
    w=torch.randn(6144,device=dev).bfloat16()
    for rows in rows_list:
        before = comm.counter.item() if rows == 0 else None
        x=torch.randn((rows,6144),device=dev).bfloat16()
        r=torch.randn_like(x)
        refq,refs,refr=gemma_add_rmsnorm_mxfp8(x,r,w,1e-6,32)
        refq=untile_codes(refq,rows).contiguous()
        q,s,rr=comm.norm_all_gather(x,r,w)
        expectq_cpu=[None]*4; expects_cpu=[None]*4
        dist.all_gather_object(expectq_cpu,refq.cpu())
        dist.all_gather_object(expects_cpu,refs.cpu())
        torch.cuda.synchronize()
        for what,a,b in [('q',q.cpu(),torch.cat(expectq_cpu)),('s',s.cpu(),torch.cat(expects_cpu)),('r',rr,refr)]:
            if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
                errors.append(f'{rank=} {rows=} {what}: {(a.view(torch.uint8)!=b.view(torch.uint8)).sum().item()} bytes')
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            comm.norm_all_gather(x,r,w,q_out=q,scale_out=s,res_out=rr)
        for rep in range(3):
            x.neg_();r.neg_()
            # Include an eager call between changed-input replays.
            eager=comm.norm_all_gather(x,r,w)
            for o in (q,s,rr):o.view(torch.uint8).fill_(0xFF)
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            for a,b in zip((q,s,rr),eager):
                if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
                    errors.append(f'{rank=} {rows=} {rep=} replay differs')
        if rows == 0 and comm.counter.item() != before:
            errors.append('zero row advanced generation')


def test_norm_gather_bits_and_changed_graphs():
    run_on_4_gpus(_worker,[0,32,96,1024,2048,4096,8192])


def _guarded_worker(rank, errors, joined):
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels.collectives.pull_collectives_ranks4.norm_gather_h6144 import NormAllGatherComm
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm import gemma_add_rmsnorm_mxfp8_row_major
    from tests.mfma_spinner import mfma_spinner
    dev = torch.device('cuda',rank)
    comm = NormAllGatherComm(rank,dist.group.WORLD,dev)
    comm.warm_up()
    compute = torch.cuda.Stream(priority=-1)
    spinner = mfma_spinner(128)
    torch.manual_seed(234+rank)
    w = torch.randn(6144,device=dev).bfloat16()
    real = MlirCompiler.compile
    count = []
    def compile_count(*a,**kw):
        count.append(1)
        return real(*a,**kw)
    for rows in (32,64,96,256,480,992,1024,1056,2048,4096,8192):
        x = torch.randn((rows,6152),device=dev).bfloat16()[:,:6144]
        r = torch.randn((rows,6160),device=dev).bfloat16()[:,:6144]
        x[:,:32]=0.;r[:,:32]=2.**-133
        x[:,32:64]=-0.;r[:,32:64]=-0.
        if rows==64:
            x[0]=2.**80 # Finite BF16 input with overflowing SSQ.
        refs = gemma_add_rmsnorm_mxfp8_row_major(x,r,w,1e-6)
        guards=[];outs=[]
        for shape,dtype in (((4*rows,6144),torch.float8_e4m3fn),((4*rows,192),torch.uint8),((rows,6144),torch.bfloat16)):
            size=shape[0]*shape[1]*dtype.itemsize
            g=torch.full((size+64,),0x5A,dtype=torch.uint8,device=dev)
            guards.append(g);outs.append(g[32:-32].view(dtype).view(shape))
        if joined:
            compute.wait_stream(torch.cuda.current_stream())
            spinner(500000,compute)
        if rank==rows%4:torch.cuda._sleep(20000)
        MlirCompiler.compile=compile_count
        try:
            comm.norm_all_gather(x,r,w,q_out=outs[0],scale_out=outs[1],res_out=outs[2])
            torch.cuda.synchronize()
        finally:
            MlirCompiler.compile=real
        if count:errors.append('compile after warm-up');count.clear()
        for i,ref in enumerate(refs):
            expected=[None]*4
            dist.all_gather_object(expected,ref.cpu().view(torch.uint8))
            want=torch.cat(expected) if i<2 else ref.cpu().view(torch.uint8)
            if not torch.equal(outs[i].cpu().view(torch.uint8),want.view(torch.uint8)):
                errors.append(f'guarded {rank=} {rows=} field {i}')
        for g in guards:
            if not bool((g[:32]==0x5A).all() & (g[-32:]==0x5A).all()):errors.append('guard overwritten')


def test_norm_gather_guards_compile_and_mfma():
    run_on_4_gpus(_guarded_worker,True)


def _reuse_worker(rank,errors):
    from amd_kernels.collectives.pull_collectives_ranks4.norm_gather_h6144 import NormAllGatherComm
    from tests.mfma_spinner import mfma_spinner
    dev=torch.device('cuda',rank)
    comm=NormAllGatherComm(rank,dist.group.WORLD,dev)
    comm.warm_up()
    rows=1024;calls=24
    torch.manual_seed(712+rank)
    sources=[torch.randn(rows,6144,device=dev).bfloat16() for _ in range(calls)]
    r=torch.randn_like(sources[0]);w=torch.randn(6144,device=dev).bfloat16()
    x=torch.empty_like(r)
    spinner=mfma_spinner(64)
    compute=torch.cuda.Stream(priority=-1)
    eager=[]
    for src in sources:
        x.copy_(src)
        eager.append(comm.norm_all_gather(x,r,w))
    graph=torch.cuda.CUDAGraph()
    captured=[]
    with torch.cuda.graph(graph):
        for src in sources:
            x.copy_(src)
            if rank==3:torch.cuda._sleep(10000)
            captured.append(comm.norm_all_gather(x,r,w))
            x.fill_(float('nan')) # Immediate producer reuse after every call.
    for rep in range(3):
        if rep==1:
            for src in sources:src.neg_()
            eager=[]
            for src in sources:
                x.copy_(src);eager.append(comm.norm_all_gather(x,r,w))
        for group in captured:
            for t in group:t.view(torch.uint8).fill_(0xFF)
        torch.cuda.synchronize()
        spinner(3000000,compute)
        graph.replay();torch.cuda.synchronize()
        for got,want in zip(captured,eager):
            for a,b in zip(got,want):
                if not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
                    errors.append(f'reuse graph {rank=} {rep=}')


def test_norm_gather_input_lifetime_long_graph():
    run_on_4_gpus(_reuse_worker)
