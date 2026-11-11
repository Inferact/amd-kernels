# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph/eager reuse, changing payloads and guarded writes for lossless gather."""
import torch
import pytest
import torch.distributed as dist

from tests.multi_gpu import run_on_4_gpus
from amd_kernels.collectives.pull_collectives_ranks4.reference import all_gather_reference


def _worker(rank, errors, num_fields):
    from amd_kernels.collectives.pull_collectives_ranks4 import AllGatherComm
    dev = torch.device('cuda', rank)
    comm = AllGatherComm(rank, dist.group.WORLD, dev, 4*2**20)
    comm.warm_up()
    compiled = [k._cf for k in (comm.stage, comm.publish, comm.pull)]
    expected_count = int(comm.counter.item())
    for rows in (1, 3, 17, 129, 352):
        inputs = [torch.empty((rows, 192), dtype=torch.uint8, device=dev),
                  torch.empty((rows, 4), dtype=torch.float32, device=dev),
                  torch.empty((rows, 4), dtype=torch.int32, device=dev),
                  torch.empty((rows,6144),dtype=torch.uint8,device=dev),
                  torch.empty((rows,192),dtype=torch.uint8,device=dev),
                  torch.empty((rows,768),dtype=torch.uint8,device=dev),
                  torch.empty((rows,16),dtype=torch.bfloat16,device=dev)][:num_fields]
        bases = [torch.empty(t.numel()*4 + 32, dtype=t.dtype, device=dev) for t in inputs]
        outputs = [b[16:-16].view(4*rows, *t.shape[1:]) for b, t in zip(bases, inputs)]
        comm.all_gather(outputs, inputs)
        expected_count += 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(3):
                comm.all_gather(outputs, inputs)
        for iteration in range(5):
            for j, t in enumerate(inputs):
                t.copy_((torch.arange(t.numel(),device=dev).view(t.shape) + rank*19 + iteration*7 + j).to(t.dtype))
                bases[j].fill_(91)
            if iteration % 2:
                comm.all_gather(outputs, inputs)
                expected_count += 1
            else:
                graph.replay()
                expected_count += 3
            torch.cuda.synchronize()
            for t, o, b in zip(inputs, outputs, bases):
                other = [None]*4
                dist.all_gather_object(other, t.cpu())
                if not torch.equal(o.cpu(), all_gather_reference(other)):
                    errors.append(f'gather mismatch r{rank} rows{rows} iteration{iteration}')
                if not (torch.all(b[:16] == 91) and torch.all(b[-16:] == 91)):
                    errors.append(f'guard mismatch r{rank} rows{rows}')
            if int(comm.counter.item()) != expected_count:
                errors.append(f'counter mismatch rank{rank}')
    assert compiled == [k._cf for k in (comm.stage, comm.publish, comm.pull)]


@pytest.mark.parametrize('num_fields',[3,6,7])
def test_graph_eager_gather_changed_payloads(num_fields):
    run_on_4_gpus(_worker, num_fields, timeout=300)


def _compact_worker(rank, errors, spin=False):
    from amd_kernels.collectives.pull_collectives_ranks4 import AllGatherComm
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInput, gather_mxfp4_moe
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda',rank)
    comm = AllGatherComm(rank,dist.group.WORLD,dev,8192*9632)
    comm.warm_up()
    if spin:
        from tests.mfma_spinner import mfma_spinner
        spinner = mfma_spinner(32)
        side = torch.cuda.Stream(priority=-1)
    compiles = []
    original_compile = MlirCompiler.compile
    for rows in (1,3,17,129,513,1025,2048,4096,4097,8192):
        x = torch.randn((rows,6144),device=dev,dtype=torch.bfloat16)
        bits = torch.arange(65536,device=dev,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        bits = bits[torch.isfinite(bits)]
        x.view(-1)[:min(x.numel(),bits.numel())].copy_(bits[:min(x.numel(),bits.numel())])
        x[-1].zero_()
        q,s = mxfp4_quant(x)
        q8,s8 = shared_quant(x)
        ids = torch.arange(rows*4,device=dev,dtype=torch.int32).view(rows,4)%128
        weights = torch.randn((rows,4),device=dev)
        source = MoeInput(q,s,weights,ids,q8,s8)
        wants = [torch.empty((4*rows,*t.shape[1:]),dtype=t.dtype,device=dev) for t in source.tensors()]
        comm.all_gather(wants,source.tensors())
        eager = gather_mxfp4_moe(source,comm).expand()
        torch.cuda.synchronize()
        for a,b in zip(eager.tensors(),wants):
            assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),(rank,rows,'eager')
        def counting(*a,**kw):
            compiles.append(1)
            return original_compile(*a,**kw)
        MlirCompiler.compile = counting
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = gather_mxfp4_moe(source,comm).expand()
            for iteration in range(3):
                for t in source.tensors():
                    t.copy_(t.roll(1,0))
                source.topk_weights.add_(.125)
                comm.all_gather(wants,source.tensors())
                for t in got.tensors():
                    t.view(torch.uint8).fill_(255)
                if spin:
                    with torch.cuda.stream(side):
                        spinner(100000,side)
                graph.replay()
                if spin:
                    torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                for i,(a,b) in enumerate(zip(got.tensors(),wants)):
                    assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),(rank,rows,i,'changed replay')
        finally:
            MlirCompiler.compile = original_compile
    assert not compiles,compiles


def test_compact_gather_matches_direct_operands():
    run_on_4_gpus(_compact_worker,timeout=300)


def test_compact_gather_with_mfma_work():
    run_on_4_gpus(_compact_worker,True,timeout=300)


def _finite_pairs_worker(rank, errors):
    """All finite BF16 value/max pairs, partitioned over the four sources."""
    from amd_kernels.collectives.pull_collectives_ranks4 import AllGatherComm
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInput, gather_mxfp4_moe
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
    dev = torch.device('cuda',rank)
    pos = torch.arange(0x7f80,device=dev,dtype=torch.int32)
    values = torch.cat((pos,pos|0x8000))
    blocks = ((values.numel()+30)//31+191)//192*192
    padded = torch.zeros(blocks*31,device=dev,dtype=torch.int32)
    padded[:values.numel()] = values
    padded = padded.reshape(1,blocks,31)
    batch = 32
    rows = batch*blocks//192
    comm = AllGatherComm(rank,dist.group.WORLD,dev,rows*9632)
    comm.warm_up()
    ids = torch.full((rows,4),rank,device=dev,dtype=torch.int32)
    weights = torch.full((rows,4),rank+.25,device=dev)
    specs = ((3072,torch.uint8),(192,torch.uint8),(4,torch.float32),(4,torch.int32),
             (6144,torch.uint8),(192,torch.uint8))
    want = [torch.empty((4*rows,k),device=dev,dtype=t) for k,t in specs]
    got = [torch.empty_like(t) for t in want]
    for first in range(0,0x7f80//4,batch):
        start = rank*(0x7f80//4)+first
        maxima = torch.arange(start,start+batch,device=dev,dtype=torch.int32)[:,None,None]
        v = torch.minimum(padded&0x7fff,maxima)|(padded&0x8000)
        raw = torch.empty((batch,blocks,32),device=dev,dtype=torch.int16)
        raw[:,:,0] = maxima[:,:,0].short()
        raw[:,:,1:] = v.short()
        x = raw.view(torch.bfloat16).reshape(rows,6144)
        q4,s4 = mxfp4_quant(x)
        q8,s8 = shared_quant(x)
        source = MoeInput(q4,s4,weights,ids,q8,s8)
        comm.all_gather(want,source.tensors())
        got = gather_mxfp4_moe(source,comm).expand().tensors()
        torch.cuda.synchronize()
        if not all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(got,want)):
            errors.append(f'finite BF16 pair mismatch r{rank} max_start{start}')
    if rank == 0:
        print('compact TP4 transport: 2,130,739,200 finite BF16 value/max slots, zero mismatches',flush=True)


def test_compact_pull_all_finite_bf16_value_max_pairs():
    run_on_4_gpus(_finite_pairs_worker,timeout=300)
