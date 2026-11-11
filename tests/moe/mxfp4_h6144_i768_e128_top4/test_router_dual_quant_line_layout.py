# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router-side shared operand matches separate quantization, including padded scales."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_quant_line_layout import router_mxfp4_quant_line_layout
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_mxfp8_quant_line_layout import router_mxfp4_mxfp8_quant_line_layout
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_dense import _shuffle_scales
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import _run_compiled
from .test_entry import weights
from .workload import make_activation

@pytest.mark.parametrize('m',[769,1024,1025,1536,2047,2048,2049,2688,2689,4096,4097,5376,5377,8192,8193,10880,10881,16384,16385,21760,21761,24576,24577,32768])
def test_dual_quant_line_layout(weights,m):
    _,w=weights;x=make_activation(m,seed=m);x[0,:64]=0
    q4,s4,parts=router_mxfp4_quant_line_layout(x,w.router_weight_lines)
    q8,s8=shared_quant(x)
    ss=torch.empty(((m+31)//32*32,192),device=x.device,dtype=torch.uint8)
    _run_compiled(_shuffle_scales(6144),s8.view(-1),ss.view(-1),m,torch.cuda.current_stream())
    expected=[q4,s4,q8,ss,parts]
    def call():
        a,b,p=router_mxfp4_mxfp8_quant_line_layout(x,w.router_weight_lines)
        return [*a,*b,p]
    actual=call()
    assert all(torch.equal(a,b) for a,b in zip(actual,expected)),m
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=call()
    for _ in range(3):
        for t in out[:-1]:t.fill_(255)
        out[-1].fill_(float('nan'))
        g.replay();torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip(out,expected)),m

@pytest.mark.parametrize('m',[257,300,513,577,640,1000,1024,2047])
def test_topk_dual_quant_s8_tiles(weights,m,monkeypatch):
    """The fused top-k + dual quant's tiled FP8 scales (padding rows zero) equal its row-major
    scales through the dense scale shuffle; everything else bit-equal, no byte outside."""
    import math
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_gemm import router_gemm_partials
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk_mxfp4_mxfp8_quant import splitk_biased_topk_mxfp4_mxfp8_quant
    _,w=weights;x=make_activation(m,seed=m);x[0,:64]=0
    x[1:9]*=torch.exp2(torch.linspace(-60,60,8,device=x.device))[:,None]
    parts=router_gemm_partials(x,w.router_weight)
    def run(tiles):
        ids=torch.empty((m,4),dtype=torch.int32,device=x.device);wt=torch.empty((m,4),dtype=torch.float32,device=x.device)
        return ids,wt,*splitk_biased_topk_mxfp4_mxfp8_quant(parts,w.correction_bias,wt,ids,w.routed_scaling_factor,x,s8_tiles=tiles)
    ids,wt,(q4,s4),(q8,s8)=run(False)
    ss=torch.empty(((m+31)//32*32,192),device=x.device,dtype=torch.uint8)
    _run_compiled(_shuffle_scales(6144),s8.view(-1),ss.view(-1),m,torch.cuda.current_stream())
    original=torch.empty;guards=[]
    def empty(*size,**kw):
        shape=tuple(size[0]) if isinstance(size[0],(tuple,list,torch.Size)) else tuple(size)
        dt=kw.get('dtype',torch.float32);g=128//torch.tensor([],dtype=dt).element_size()
        storage=original((math.prod(shape)+2*g,),**kw);storage.fill_(19)
        out=storage[g:-g].view(shape);out.fill_(255 if dt==torch.uint8 else float('nan') if dt.is_floating_point else -7)
        guards.append((storage,g));return out
    with monkeypatch.context() as mp:
        mp.setattr(torch,'empty',empty)
        got=run(True)
    torch.cuda.synchronize()
    i2,w2,(a4,b4),(a8,b8)=got
    assert b8.shape==ss.shape
    for a,b in zip([i2,w2,a4,b4,a8,b8],[ids,wt,q4,s4,q8,ss]):
        assert torch.equal(a,b),m
    assert all((t[:g]==19).all() and (t[-g:]==19).all() for t,g in guards)


@pytest.mark.parametrize('m',[1025,2049,4097,8193,16385,32769])
def test_dual_quant_guards_and_wide_exponents(weights,m,monkeypatch):
    import math
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import shared_mxfp8_quant_ref
    _,w=weights;x=make_activation(m,seed=m)
    x[:16]*=torch.exp2(torch.linspace(-96,96,16,device=x.device))[:,None]
    x[0]=0
    refq,refs=shared_mxfp8_quant_ref(x)
    original=torch.empty;guards=[]
    def empty(*size,**kw):
        shape=tuple(size[0]) if isinstance(size[0],(tuple,list,torch.Size)) else tuple(size)
        dt=kw.get('dtype',torch.float32);g=128//torch.tensor([],dtype=dt).element_size()
        storage=original((math.prod(shape)+2*g,),**kw);storage.fill_(19)
        out=storage[g:-g].view(shape);out.fill_(255 if dt==torch.uint8 else float('nan'))
        guards.append((storage,g));return out
    monkeypatch.setattr(torch,'empty',empty)
    _,(q,s),_=router_mxfp4_mxfp8_quant_line_layout(x,w.router_weight_lines)
    torch.cuda.synchronize()
    assert torch.equal(q,refq)
    # Independent torch inverse of the dense scale permutation (incl. padding zeros).
    row=s.view(-1,24,4,16,2,2).permute(0,5,3,1,4,2).reshape(-1,192)
    assert torch.equal(row[:m],refs) and not row[m:].any()
    assert all((t[:g]==19).all() and (t[-g:]==19).all() for t,g in guards)

@pytest.mark.parametrize('m',[1025,2048,4097,8192,32768])
@pytest.mark.parametrize('shared_scales_tiled',[False,True])
def test_dual_quant_next_to_mfma(weights,m,shared_scales_tiled):
    from tests.mfma_spinner import mfma_spinner
    _,w=weights;x=make_activation(m,seed=m)
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_mxfp8_quant_row_scales import router_mxfp4_mxfp8_quant_row_scales
    quant = router_mxfp4_mxfp8_quant_line_layout if shared_scales_tiled else router_mxfp4_mxfp8_quant_row_scales
    def call():
        a,b,p=quant(x,w.router_weight_lines)
        return [*a,*b,p]
    ref=[t.clone() for t in call()];torch.cuda.synchronize()
    spinner=mfma_spinner(256);stream=torch.cuda.Stream(priority=-1)
    spinner(5_000_000,stream)
    out=call();torch.cuda.synchronize()
    assert all(torch.equal(a,b) for a,b in zip(out,ref))
