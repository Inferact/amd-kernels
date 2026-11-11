# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest, torch
from aiter.ops.shuffle import shuffle_scale_a16w4
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8,warmup_gemma_rmsnorm_mxfp8
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144.reference import gemma_rmsnorm_reference,quantize_mxfp8_reference as quantize_mxfp8

ROWS=(0,1,7,8,13,70,80,112,128,224,256,448,512,896,1024,1792,2048)
TILES=(0,16,32,64,96,128)

def unpack(q,s,m,tile):
    qq=q.view(torch.uint8)
    if tile:
        qq=qq.reshape(-1,96,4,16,16).permute(0,3,1,2,4).reshape(-1,6144)
    row=torch.arange(s.shape[0],device=s.device)[:,None].expand(s.shape[0],192)
    col=torch.arange(192,device=s.device)[None,:].expand_as(row)
    off=(((((row//32)*24+col//8)*4+col%4)*16+row%16)*2+(col%8)//4)*2+(row%32)//16
    return qq[:m],s.reshape(-1)[off[:m]]

@pytest.mark.parametrize('m',ROWS)
def test_contract(m):
    torch.manual_seed(m+17)
    x=(torch.randn((m,6160),device='cuda')*torch.exp2(torch.randint(-5,6,(m,1),device='cuda').float())).bfloat16()[:,:6144]
    if m:x[0].zero_()
    w=(torch.randn(6144,device='cuda')*.3).bfloat16()
    ref=gemma_rmsnorm_reference(x,w,1e-6)
    for tile in TILES:
        out=torch.empty((m,6144),device='cuda',dtype=torch.bfloat16)
        q,s=gemma_rmsnorm_mxfp8(x,w,1e-6,tile,bf16_out=out)
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        if m>1:
            assert torch.nn.functional.cosine_similarity(out[1:].double(),ref[1:].double(),dim=1).min()>=.999999
        rq,rs=quantize_mxfp8(out)
        uq,us=unpack(q,s,m,tile)
        assert torch.equal(uq,rq.view(torch.uint8))
        assert torch.equal(us,rs)
        qa,sa=gemma_rmsnorm_mxfp8(x,w,1e-6,tile)
        aq,ass=unpack(qa,sa,m,tile)
        assert torch.equal(aq,uq) and torch.equal(ass,us)

@pytest.mark.parametrize('tile',TILES)
def test_graph_poison_change_and_repeat(tile):
    m=70
    x=torch.randn((m,6144),device='cuda',dtype=torch.bfloat16);w=x[0].clone()
    gemma_rmsnorm_mxfp8(x,w,1e-6,tile)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):q,s=gemma_rmsnorm_mxfp8(x,w,1e-6,tile)
    for i in range(3):
        x.mul_(.5).add_(.125)
        eq,es=gemma_rmsnorm_mxfp8(x,w,1e-6,tile)
        q.view(torch.uint8).fill_(255);s.fill_(255)
        g.replay();torch.cuda.synchronize()
        a,b=unpack(q,s,m,tile);c,d=unpack(eq,es,m,tile)
        assert torch.equal(a,c) and torch.equal(b,d)

def test_warmup():
    from flydsl.compiler.jit_function import MlirCompiler
    warmup_gemma_rmsnorm_mxfp8()
    orig=MlirCompiler.compile
    def unexpected(*a,**kw):raise AssertionError('compile after warm-up')
    MlirCompiler.compile=unexpected
    try:
        for m in ROWS:
            x=torch.zeros((m,6144),device='cuda',dtype=torch.bfloat16);w=torch.zeros(6144,device='cuda',dtype=torch.bfloat16)
            for tile in TILES:
                for out in (None,torch.empty_like(x)):
                    gemma_rmsnorm_mxfp8(x,w,1e-6,tile,bf16_out=out)
        torch.cuda.synchronize()
    finally:MlirCompiler.compile=orig


def test_mfma_co_residency():
    from tests.mfma_spinner import mfma_spinner
    spinner=mfma_spinner(256)
    side=torch.cuda.Stream(priority=-1)
    for m in (7,224,896,1792):
        x=torch.randn((m,6144),device='cuda',dtype=torch.bfloat16);w=x[0].clone()
        ref=torch.empty_like(x)
        gemma_rmsnorm_mxfp8(x,w,1e-6,16,bf16_out=ref)
        torch.cuda.synchronize()
        for _ in range(3):
            got=torch.empty_like(x)
            spinner(100_000,side)
            gemma_rmsnorm_mxfp8(x,w,1e-6,16,bf16_out=got)
            torch.cuda.synchronize()
            assert torch.equal(ref,got),m
