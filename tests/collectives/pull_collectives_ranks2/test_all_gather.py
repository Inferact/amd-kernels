# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: graph/eager reuse, changing payloads and guarded writes for lossless gather; the compact MoE payload
through the i1536 MoE's gather."""
import torch
import pytest
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus
from amd_kernels.collectives.pull_collectives_ranks2.reference import all_gather_reference


def _worker(rank, errors, num_fields):
    from amd_kernels.collectives.pull_collectives_ranks2 import AllGatherComm
    dev = torch.device('cuda', rank)
    comm = AllGatherComm(rank, dist.group.WORLD, dev, 4*2**20)
    comm.warm_up()
    compiled = comm.kernel._cf
    expected_count = int(comm.counter.item())
    per_call = -(-num_fields // 6)  # one launch per six fields
    for rows in (1, 3, 17, 129, 352):
        inputs = [torch.empty((rows, 192), dtype=torch.uint8, device=dev),
                  torch.empty((rows, 4), dtype=torch.float32, device=dev),
                  torch.empty((rows, 4), dtype=torch.int32, device=dev),
                  torch.empty((rows,6144),dtype=torch.uint8,device=dev),
                  torch.empty((rows,192),dtype=torch.uint8,device=dev),
                  torch.empty((rows,768),dtype=torch.uint8,device=dev),
                  torch.empty((rows,16),dtype=torch.bfloat16,device=dev)][:num_fields]
        bases = [torch.empty(t.numel()*2 + 32, dtype=t.dtype, device=dev) for t in inputs]
        outputs = [b[16:-16].view(2*rows, *t.shape[1:]) for b, t in zip(bases, inputs)]
        comm.all_gather(outputs, inputs)
        expected_count += per_call
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
                expected_count += per_call
            else:
                graph.replay()
                expected_count += 3 * per_call
            torch.cuda.synchronize()
            for t, o, b in zip(inputs, outputs, bases):
                other = [None]*2
                dist.all_gather_object(other, t.cpu())
                if not torch.equal(o.cpu(), all_gather_reference(other)):
                    errors.append(f'gather mismatch r{rank} rows{rows} iteration{iteration}')
                if not (torch.all(b[:16] == 91) and torch.all(b[-16:] == 91)):
                    errors.append(f'guard mismatch r{rank} rows{rows}')
            if int(comm.counter.item()) != expected_count:
                errors.append(f'counter mismatch rank{rank}')
    assert compiled is comm.kernel._cf


@pytest.mark.parametrize('num_fields',[3,6,7])
def test_graph_eager_gather_changed_payloads(num_fields):
    run_on_2_gpus(_worker, num_fields, timeout=300)


def _compact_worker(rank, errors, spin=False):
    from amd_kernels.collectives.pull_collectives_ranks2 import AllGatherComm
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeInput, gather_mxfp4_moe
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.quant import mxfp4_quant
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4.shared_quant import shared_quant
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda',rank)
    comm = AllGatherComm(rank,dist.group.WORLD,dev,16384*9632)
    comm.warm_up()
    if spin:
        from tests.mfma_spinner import mfma_spinner
        spinner = mfma_spinner(32)
        side = torch.cuda.Stream(priority=-1)
    compiles = []
    original_compile = MlirCompiler.compile
    for rows in (1,3,17,129,513,1025,2048,4096,4097,8192,16384):
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
        wants = [torch.empty((2*rows,*t.shape[1:]),dtype=t.dtype,device=dev) for t in source.tensors()]
        comm.all_gather(wants,source.tensors())
        eager = gather_mxfp4_moe(source,comm,world=2).expand()
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
                got = gather_mxfp4_moe(source,comm,world=2).expand()
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
    run_on_2_gpus(_compact_worker,timeout=300)


def test_compact_gather_with_mfma_work():
    run_on_2_gpus(_compact_worker,True,timeout=300)


def _compact_bytes_worker(rank, errors, rows_list):
    """The six compact MoE fields (7328 B/row) as raw bytes at the served shard rows, every byte checked, both the
    joined six-field stage + publish and the separate stages, eager and replayed."""
    from amd_kernels.collectives.pull_collectives_ranks2 import AllGatherComm
    from flydsl.compiler.jit_function import MlirCompiler
    dev = torch.device('cuda', rank)
    comm = AllGatherComm(rank, dist.group.WORLD, dev, 16384*9632)
    comm.warm_up()
    real = MlirCompiler.compile
    compiles = []
    def counting(*a, **kw):
        compiles.append(1)
        return real(*a, **kw)
    cols = (192, 16, 16, 6144, 192, 768)
    for rows in rows_list:
        every = []
        for p in range(2):
            g = torch.Generator(device=dev).manual_seed(rows * 7 + p)
            every.append([torch.randint(0, 256, (rows, c), generator=g, dtype=torch.uint8, device=dev) for c in cols])
        outs = [torch.empty((2*rows, c), dtype=torch.uint8, device=dev) for c in cols]
        MlirCompiler.compile = counting
        try:
            comm.all_gather(outs, every[rank])
            torch.cuda.synchronize()
            for i, o in enumerate(outs):
                if not torch.equal(o, all_gather_reference([e[i] for e in every])):
                    errors.append(f'rank {rank} rows {rows} field {i} eager')
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                comm.all_gather(outs, every[rank])
            for t in every[rank]:
                t.copy_(t.roll(1, 0))
            for p in range(2):
                if p != rank:
                    for t in every[p]:
                        t.copy_(t.roll(1, 0))
            for o in outs:
                o.fill_(7)
            graph.replay()
            torch.cuda.synchronize()
            for i, o in enumerate(outs):
                if not torch.equal(o, all_gather_reference([e[i] for e in every])):
                    errors.append(f'rank {rank} rows {rows} field {i} replay')
        finally:
            MlirCompiler.compile = real
        del graph
    if compiles:
        errors.append(f'{rank}: {len(compiles)} compiles after warm-up')


def test_compact_fields_served_rows():
    # the part count per field follows its bytes: sizes rise and fall on one communicator
    run_on_2_gpus(_compact_bytes_worker, [32, 1024, 2048, 4096, 4128, 8192, 16384, 1024, 16384, 32, 2048], timeout=600)
