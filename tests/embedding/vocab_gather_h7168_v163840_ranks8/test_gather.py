# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import torch
import torch.distributed as dist
from amd_kernels.embedding.vocab_gather_h7168_v163840_ranks8 import VocabGather
from amd_kernels.embedding.vocab_gather_h7168_v163840_ranks8.reference import vocab_gather_reference
from tests.multi_gpu8 import run_on_8_gpus


def worker(rank,errors):
    from flydsl.compiler.jit_function import MlirCompiler
    dev=torch.device('cuda',rank)
    torch.manual_seed(37)
    shards=[torch.randn((128,7168),dtype=torch.bfloat16) for _ in range(8)]
    for shard in shards:
        shard[0,:8]=torch.tensor([0.,-0.,2.**-133,-2.**-133,2.**-126,-2.**-126,1.,-1.],dtype=torch.bfloat16)
    weight=shards[rank].to(dev)
    weight.view(torch.int16).bitwise_xor_(0)  # GPU writes before load-time publication
    c=VocabGather(weight,rank,dist.group.WORLD)
    c.warm_up()
    orig=MlirCompiler.compile
    def unexpected(*a,**kw):raise AssertionError('compile after warmup')
    MlirCompiler.compile=unexpected
    for test_case in range(18):
        m=(1,12,40,65,80,128,192,256,2048)[test_case%9]
        dtype=torch.int32 if test_case<9 else torch.int64
        ids=torch.randint(-2,1026,(m,),dtype=dtype)
        ids[0]=0
        if m>9:ids[:9]=torch.tensor([0,128,256,384,-1,512,511,1023,1024])
        inputs=ids.to(dev)
        expected=vocab_gather_reference(shards,ids).to(dev)
        eager=c(inputs)
        assert torch.equal(eager.view(torch.uint8),expected.view(torch.uint8)),('eager',rank,m)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=c(inputs)
        for _ in range(3):
            out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8)),('graph',rank,m)
        new_ids=(ids+17).to(dev)
        inputs.copy_(new_ids);graph.replay();torch.cuda.synchronize()
        new_ref=vocab_gather_reference(shards,new_ids.cpu()).to(dev)
        assert torch.equal(out.view(torch.uint8),new_ref.view(torch.uint8)),('changed',rank,m)
    for m in (12,80,128,192):
        for name,ids in (('same token',torch.full((m,),5)),
                         ('one owner',3*128+torch.randint(0,128,(m,))),
                         ('balanced owners',(torch.arange(m)%8)*128+torch.randint(0,128,(m,))),
                         ('descending',torch.arange(1023,1023-m,-1)%1024)):
            inputs=ids.to(dev)
            expected=vocab_gather_reference(shards,ids).to(dev)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=c(inputs)
            out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
            assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8)),(name,rank,m)
    MlirCompiler.compile=orig
    c.close()


def test_peer_embedding():run_on_8_gpus(worker,timeout=180)
