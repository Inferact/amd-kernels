# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import torch
import torch.distributed as dist
from amd_kernels.embedding.vocab_gather_h6144 import VocabGather
from amd_kernels.embedding.vocab_gather_h6144.reference import vocab_gather_reference
from tests.multi_gpu import run_on_4_gpus


def worker(rank,errors):
    from flydsl.compiler.jit_function import MlirCompiler
    dev=torch.device('cuda',rank)
    torch.manual_seed(37)
    shards=[torch.randn((128,6144),dtype=torch.bfloat16) for _ in range(4)]
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
        m=(1,7,70,112,224,448,896,1792,2048)[test_case%9]
        dtype=torch.int32 if test_case<9 else torch.int64
        ids=torch.randint(-2,514,(m,),dtype=dtype)
        ids[0]=0
        if m>7:ids[:7]=torch.tensor([0,128,256,384,-1,512,511])
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
    MlirCompiler.compile=orig
    c.close()


def test_peer_embedding():run_on_4_gpus(worker,timeout=180)
