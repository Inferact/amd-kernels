# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Peer lookup + plain norm + BF16-rule quant, including raw residual and changed graph IDs."""
import torch
import torch.distributed as dist
from amd_kernels.embedding.vocab_gather_h6144 import VocabGather
from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import vocab_gather_gemma_mxfp8,warmup_vocab_gather_gemma_mxfp8
from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144.reference import lookup_reference
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8
from tests.multi_gpu import run_on_4_gpus


def written(q,s,m,tile):
    row=torch.arange(m,device=q.device)[:,None]
    col=torch.arange(6144,device=q.device)[None,:]
    group=col[:,:192]
    off=((row//16)*(16*6144)+(col//64)*1024+(col%64//16)*256+(row%16)*16+col%16
         if tile else row*6144+col)
    so=((((row//32)*24+group//8)*4+group%4)*16+row%16)*4+(group%8//4)*2+row%32//16
    return q.view(torch.uint8).flatten()[off],s.flatten()[so]


def worker(rank,errors):
    from flydsl.compiler.jit_function import MlirCompiler
    from tests.mfma_spinner import mfma_spinner
    torch.set_num_threads(8)
    dev=torch.device('cuda',rank)
    gen=torch.Generator().manual_seed(963)
    shards=[(torch.randn((128,6144),generator=gen)*torch.exp2(torch.randint(-5,6,(128,192,1),generator=gen).expand(128,192,32).reshape(128,6144).float())).bfloat16() for _ in range(4)]
    for shard in shards:
        shard[0].zero_()
        shard[1,:8]=torch.tensor([0.,-0.,2.**-133,-2.**-133,2.**-126,-2.**-126,1.,-1.],dtype=torch.bfloat16)
    weight=shards[rank].to(dev)
    weight.view(torch.int16).bitwise_xor_(0) # write on GPU before owner publication
    lookup=VocabGather(weight,rank,dist.group.WORLD)
    norm=torch.randn(6144,generator=gen).bfloat16().to(dev)
    warmup_vocab_gather_gemma_mxfp8(lookup,norm)
    # Warm both constituent baseline paths before trapping compiles.
    for m in (16,513):
        for dtype in (torch.int32,torch.int64):
            ids=torch.zeros(m,dtype=dtype,device=dev)
            for tile in (0,16):gemma_rmsnorm_mxfp8(lookup(ids),norm,1e-6,tile)
    spinner=mfma_spinner(256);side=torch.cuda.Stream(priority=-1)
    orig=MlirCompiler.compile
    def unexpected(*a,**kw):raise AssertionError('compile after warmup')
    MlirCompiler.compile=unexpected
    try:
        for dtype in (torch.int32,torch.int64):
            for tile in (0,16):
                for m in (0,1,7,33,70,112,224,448,512,513,896,1792,2048):
                    ids=torch.randint(-2,514,(m,),generator=gen,dtype=dtype)
                    if m:ids[0]=0
                    if m>7:ids[:7]=torch.tensor([0,1,128,256,384,-1,512])
                    inputs=ids.to(dev)
                    raw=lookup(inputs)
                    q,s=gemma_rmsnorm_mxfp8(raw,norm,1e-6,tile)
                    a,b,r=vocab_gather_gemma_mxfp8(inputs,lookup,norm,1e-6,tile)
                    expected=lookup_reference(ids,shards).to(dev)
                    assert torch.equal(r.view(torch.int16),expected.view(torch.int16)),('raw',rank,m,dtype,tile)
                    x,y=written(q,s,m,tile);xx,yy=written(a,b,m,tile)
                    assert torch.equal(x,xx) and torch.equal(y,yy),('quant',rank,m,dtype,tile)
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):ao,bo,ro=vocab_gather_gemma_mxfp8(inputs,lookup,norm,1e-6,tile)
                    for _ in range(2):
                        ao.view(torch.uint8).fill_(0x7f);bo.fill_(0xff);ro.fill_(float('nan'))
                        graph.replay();torch.cuda.synchronize()
                        ax,bx=written(ao,bo,m,tile)
                        assert torch.equal(ax,x) and torch.equal(bx,y) and torch.equal(ro,r)
                    inputs.add_(17)
                    fresh=vocab_gather_gemma_mxfp8(inputs,lookup,norm,1e-6,tile)
                    graph.replay();torch.cuda.synchronize()
                    ax,bx=written(ao,bo,m,tile);fx,fy=written(*fresh[:2],m,tile)
                    assert torch.equal(ax,fx) and torch.equal(bx,fy) and torch.equal(ro,fresh[2])
                # Scalar overflow must not wrap into the vocabulary.
                if dtype==torch.int64:
                    ids=torch.tensor([2**33,-2**33],device=dev)
                    a,b,r=vocab_gather_gemma_mxfp8(ids,lookup,norm,code_tile_rows=tile)
                    assert not r.count_nonzero()
                    aq,_=written(a,b,2,tile);assert not (aq&0x7f).count_nonzero()
        # Same rows have identical codes on both physical-wave variants and layouts.
        base=None
        for m in (7,224,1792):
            ids=torch.full((m,),19,device=dev,dtype=torch.int32)
            for tile in (0,16):
                a,b,_=vocab_gather_gemma_mxfp8(ids,lookup,norm,code_tile_rows=tile)
                aq,bs=written(a,b,m,tile)
                if base is None:base=(aq[0].clone(),bs[0].clone())
                assert torch.equal(aq,base[0].expand_as(aq)) and torch.equal(bs,base[1].expand_as(bs))
        for m in (7,896):
            ids=torch.arange(m,device=dev,dtype=torch.int32)%512
            ref=vocab_gather_gemma_mxfp8(ids,lookup,norm)
            torch.cuda.synchronize()
            spinner(100_000,side)
            got=vocab_gather_gemma_mxfp8(ids,lookup,norm)
            torch.cuda.synchronize()
            a,b=written(*ref[:2],m,16);x,y=written(*got[:2],m,16)
            assert torch.equal(a,x) and torch.equal(b,y) and torch.equal(ref[2],got[2])
    finally:MlirCompiler.compile=orig
    lookup.close()


def test_peer_lookup_norm_quant():run_on_4_gpus(worker,timeout=300)
