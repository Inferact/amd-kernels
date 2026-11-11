# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Independent top-k/quant tasks retain routing ties, quant bits and replay writes."""
import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk import splitk_biased_topk
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import mxfp4_quant_ref
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk_mxfp4_mxfp8_quant_sorted_tasks import splitk_biased_topk_mxfp4_mxfp8_quant as sorted_tasks
from .workload import require_gfx950, make_activation


@pytest.mark.parametrize('m',[1,3,8,12,13,16,17,32,64,65,96,128,192,193,256,257,384,385,512,513,769,1024])
def test_bits_guards_changed_graph(m,monkeypatch):
    require_gfx950()
    g=torch.Generator(device='cuda').manual_seed(m)
    split=48 if m<=64 else 16
    # Non-dense token/split strides and top-k row strides exercise pointer bounds.
    storage=torch.randn(m,split,132,device='cuda',generator=g)
    part=storage[:,:,:128]
    bias=torch.randn(128,device='cuda',generator=g)*.1
    x=make_activation(m,seed=m)
    x[0,:32]=0
    wt=torch.full((m,8),float('nan'),device='cuda')[:,1:5]
    ids=torch.full((m,8),-1,device='cuda',dtype=torch.int32)[:,1:5]
    original=torch.empty;guards=[]
    def alloc(*args,**kwargs):
        shape=tuple(args[0]) if isinstance(args[0],(tuple,list)) else tuple(args)
        total=1
        for n in shape:total*=n
        t=torch.full((total+128,),0xA5,device=kwargs['device'],dtype=kwargs['dtype'])
        guards.append(t)
        return t[64:-64].view(shape)
    def reference():
        splitk_biased_topk(part,bias,wt,ids,2.0)
        q,s=mxfp4_quant_ref(x)
        q8,s8=shared_quant(x)
        return [wt.clone(),ids.clone(),q,s,q8,s8]
    def run(fn):
        a,b=fn(part,bias,wt,ids,2.,x)
        return [wt,ids,*a,*b]
    expected=reference()
    monkeypatch.setattr(torch,'empty',alloc)
    actual=run(sorted_tasks)
    assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(actual,expected))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run(sorted_tasks)
    monkeypatch.setattr(torch,'empty',original)
    for change in range(3):
        if change:
            part.copy_(torch.randn(part.shape,device='cuda',generator=g))
            x.copy_(make_activation(m,seed=m+change))
        if change==2:
            part.zero_();bias.zero_()  # all experts tied, established strict tree
        expected=reference()
        for t in out:t.fill_(0 if t.dtype==torch.uint8 else -1)
        graph.replay();torch.cuda.synchronize()
        assert all(torch.equal(a.view(torch.uint8),b.view(torch.uint8)) for a,b in zip(out,expected))
        q,s=mxfp4_quant_ref(x)
        assert torch.equal(out[2],q) and torch.equal(out[3],s)
        q8,s8=shared_quant(x)
        assert torch.equal(out[4],q8) and torch.equal(out[5],s8)
        assert all((t[:64]==0xA5).all() and (t[-64:]==0xA5).all() for t in guards)
    assert (wt._base[:,0].isnan()).all() and (wt._base[:,5:].isnan()).all()
    assert (ids._base[:,0]==-1).all() and (ids._base[:,5:]==-1).all()
