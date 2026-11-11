# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Mixed-format persistent launch: independent oracle, changed graphs and metadata reuse."""
import importlib
import pytest
import torch
from .test_shared_expert import weights,complete_weights
from .workload import make_activation,small_slots
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import shared_mxfp8_ref

PKG='amd_kernels.moe.mxfp4_h6144_i768_e128_top4'

def old_entry(x,w,monkeypatch):
    """The routed and shared GEMMs in separate launches: at 449-512 rows the small-tile host's routed slots and the
    standalone shared expert, elsewhere the entry with the combined launch switched off."""
    if 448<x.shape[0]<=512:
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import route
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert import shared_expert
        return small_slots(x,w,*route(x,w))[0],shared_expert(x,w.shared_weights)
    sh=importlib.import_module(PKG+'.shared_expert')
    with monkeypatch.context() as mp:
        mp.setattr(sh,'shared_in_persistent_for',lambda n:False)
        return mxfp4_moe(x,w,return_partials=True)


@pytest.mark.parametrize('m',[448,449,450,480,511,512,513,577,704,705,768,769,1000,1024,1025,1088,1280,1408,1535,1536,1537,1792,2047,2048,2049,
                              2560,3072,3583,3584,3840,4032,4095,4096])
def test_shared_persistent_entry(complete_weights,weights,m,monkeypatch):
    sh=importlib.import_module(PKG+'.shared_expert')
    es=importlib.import_module(PKG+'.experts_a4w4_bm128')
    enabled=sh.shared_in_persistent_for(m)
    w=complete_weights;x=make_activation(m,seed=8074+m)
    expected=old_entry(x,w,monkeypatch)
    fn=es._gemm12_shared;calls=[]
    def counted(*args):
        launch=fn(*args);calls.append(launch.kernel_name);return launch
    monkeypatch.setattr(es,'_gemm12_shared',counted)
    out=mxfp4_moe(x,w,return_partials=True)
    torch.cuda.synchronize()
    assert bool(calls)==enabled
    assert all('gemm12_a4w4_shared_a8w8' in name for name in calls)
    assert all(torch.equal(a,b) and torch.isfinite(a).all() for a,b in zip(out,expected))
    for _ in range(2):
        again=mxfp4_moe(x,w,return_partials=True)
        assert all(torch.equal(a,b) for a,b in zip(out,again))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):captured=mxfp4_moe(x,w,return_partials=True)
    for seed in [74,75]:
        x.copy_(make_activation(m,seed=seed))
        expected=old_entry(x,w,monkeypatch)
        for t in captured:t.fill_(float('nan'))
        torch.cuda.synchronize();g.replay();torch.cuda.synchronize()
        assert all(torch.equal(a,b) for a,b in zip(captured,expected))
        assert all(torch.count_nonzero(t).item()==0 for t in es._ready.values())
    raw,_=weights
    ref=shared_mxfp8_ref(x[:32],*raw)
    assert torch.nn.functional.cosine_similarity(captured[1][:32].float(),ref,dim=1).min()>=.999
    reduced=mxfp4_moe(x,w)
    acc=torch.zeros_like(captured[1],dtype=torch.float32)
    for k in range(4):acc=acc+captured[0][:,k].float()
    assert torch.equal(reduced,(acc+captured[1].float()).bfloat16())


@pytest.mark.parametrize('m',[449,512,513,768,769,1024,1025,1536,2048,3072,4096])
def test_shared_persistent_mfma_corunner(complete_weights,m,monkeypatch):
    from tests.mfma_spinner import mfma_spinner
    es=importlib.import_module(PKG+'.experts_a4w4_bm128')
    x=make_activation(m,seed=m);w=complete_weights
    expected=old_entry(x,w,monkeypatch)
    torch.cuda.synchronize()
    spinner=mfma_spinner(256);stream=torch.cuda.Stream(priority=-1)
    spinner(5_000_000,stream)
    out=mxfp4_moe(x,w,return_partials=True)
    torch.cuda.synchronize()
    assert all(torch.equal(a,b) for a,b in zip(out,expected))
    assert all(torch.count_nonzero(t).item()==0 for t in es._ready.values())


@pytest.mark.parametrize('m',[449,512,513,768,769,1024,1025])
def test_shared_persistent_router_path(complete_weights,m,monkeypatch):
    """The combined launch's tiled shared scales come from the split-K top-k below
    ROUTER_QUANT_FROM rows and from the line-layout router from there."""
    ch=importlib.import_module(PKG+'.chain')
    ll=importlib.import_module(PKG+'.router_mxfp4_mxfp8_quant_line_layout')
    rt=importlib.import_module(PKG+'.router_topk_mxfp4_mxfp8_quant')
    calls=[]
    def counted(name,fn):
        def f(*args,**kw):
            calls.append((name,kw.get('s8_tiles')));return fn(*args,**kw)
        return f
    monkeypatch.setattr(ll,'router_mxfp4_mxfp8_quant_line_layout',counted('line',ll.router_mxfp4_mxfp8_quant_line_layout))
    monkeypatch.setattr(rt,'splitk_biased_topk_mxfp4_mxfp8_quant',counted('splitk',rt.splitk_biased_topk_mxfp4_mxfp8_quant))
    mxfp4_moe(make_activation(m,seed=m),complete_weights,return_partials=True)
    assert calls==([('splitk',True)] if m<ch.ROUTER_QUANT_FROM else [('line',None)])
