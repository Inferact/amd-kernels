# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Lossless transport of canonical direct-BF16 FP4/FP8 operands."""
from dataclasses import replace

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import (
    MoeInput, MoeInputCompact, mxfp4_moe, prepare_mxfp4_moe,
)
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_pack import pack_mxfp4_sidecar
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_unpack import unpack_mxfp4_sidecar
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import (
    mxfp4_sidecar_pack_ref, mxfp4_sidecar_unpack_ref,
)
from .workload import make_activation, require_gfx950, replay_check
from .test_shared_expert import weights, complete_weights


@pytest.mark.parametrize('m',[1,8,31,32,33,512,1025,2048,4097,8192,32768])
def test_sidecar_contract_and_changed_graph(m):
    require_gfx950()
    x=make_activation(m,seed=700+m)
    x[0,:32]=torch.tensor([6.,3.484375,-3.484375,3.515625,-3.515625,0.,-0.]+[0.]*25,device='cuda').bfloat16()
    q4,s4=mxfp4_quant(x);q8,s8=shared_quant(x)
    mask=pack_mxfp4_sidecar(q4,s4,q8,s8)
    out=unpack_mxfp4_sidecar(mask,s4,q8,s8)
    assert torch.equal(mask,mxfp4_sidecar_pack_ref(q4,s4,q8,s8))
    assert torch.equal(out,q4)
    assert torch.equal(out,mxfp4_sidecar_unpack_ref(mask,s4,q8,s8))
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gm=pack_mxfp4_sidecar(q4,s4,q8,s8)
        go=unpack_mxfp4_sidecar(gm,s4,q8,s8)
    for seed in [1700+m,2700+m]:
        changed=make_activation(m,seed=seed)
        operands=(*mxfp4_quant(changed),*shared_quant(changed))
        for t,v in zip((q4,s4,q8,s8),operands):t.copy_(v)
        gm.fill_(173);go.fill_(173);torch.cuda.synchronize()
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(go,q4)
        assert torch.equal(gm,mxfp4_sidecar_pack_ref(q4,s4,q8,s8))
        graph.replay();torch.cuda.synchronize()
        assert torch.equal(go,q4)


@pytest.mark.parametrize('producer',['canonical','small_prepared','row_prepared'])
def test_sidecar_all_finite_bf16_value_max_pairs(producer):
    """Every finite BF16 magnitude as block max (positive, lane 0: it sets both canonical scales), every signed source
    <= that max in the other lanes, zero and subnormals included."""
    require_gfx950()
    pos=torch.arange(0x7f80,dtype=torch.int32,device='cuda')
    values=torch.cat((pos,pos|0x8000))
    blocks=((values.numel()+30)//31+191)//192*192
    padded=torch.zeros((blocks*31,),dtype=torch.int32,device='cuda')
    padded[:values.numel()]=values
    padded=padded.reshape(1,blocks,31)
    batch=128 if producer=='row_prepared' else 32
    if producer!='canonical':
        from types import SimpleNamespace
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
        gate=torch.zeros((128,6144),device='cuda')
        w=SimpleNamespace(router_weight=gate,router_weight_lines=router_line_layout(gate),
                          correction_bias=torch.zeros(128,device='cuda'),
                          routed_scaling_factor=2.,shared_weights=True)
    for first in range(0,0x7f80,batch):
        maxima=torch.arange(first,min(first+batch,0x7f80),dtype=torch.int32,device='cuda')[:,None,None]
        v=torch.minimum(padded&0x7fff,maxima)|(padded&0x8000)
        raw=torch.empty((len(maxima),blocks,32),dtype=torch.int16,device='cuda')
        raw[:,:,0]=maxima[:,:,0].short();raw[:,:,1:]=v.short()
        x=raw.view(torch.bfloat16).reshape(-1,6144)
        q4,s4=mxfp4_quant(x);q8,s8=shared_quant(x)
        if producer!='canonical':
            p=prepare_mxfp4_moe(x,w)
            assert all(torch.equal(a,b) for a,b in zip((q4,s4,q8,s8),
                (p.routed_codes,p.routed_scales,p.shared_codes,p.shared_scales))),(producer,first)
            q4,s4,q8,s8=p.routed_codes,p.routed_scales,p.shared_codes,p.shared_scales
        mask=pack_mxfp4_sidecar(q4,s4,q8,s8)
        out=unpack_mxfp4_sidecar(mask,s4,q8,s8)
        assert torch.equal(out,q4),first


@pytest.mark.parametrize('m',[8,256,257,1025,2048,4097,8192,16384,32768])
def test_compact_public_entry(complete_weights,m,monkeypatch):
    w=complete_weights;x=make_activation(m,seed=m)
    prepared=prepare_mxfp4_moe(x,w);compact=prepared.compact()
    assert sum(t.numel()*t.element_size() for t in compact.tensors())==7328*m
    assert sum(t.numel()*t.element_size() for t in prepared.tensors())==9632*m
    restored=compact.expand()
    assert all(torch.equal(a,b) for a,b in zip(prepared.tensors(),restored.tensors()))
    expected=mxfp4_moe(prepared,w,return_partials=True)
    def fail(*args,**kw):raise AssertionError('compact entry rerouted or requantized')
    monkeypatch.setattr('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain._prepare',fail)
    monkeypatch.setattr('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant.mxfp4_quant',fail)
    eager,replay=replay_check(lambda:list(mxfp4_moe(compact,w,return_partials=True)))
    assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(expected,eager,replay))
    assert torch.equal(mxfp4_moe(compact,w),mxfp4_moe(prepared,w))


@pytest.mark.parametrize('rows',[3,129,513,2049,8192])
def test_compact_arbitrary_shard_boundaries(complete_weights,rows):
    w=complete_weights;x=make_activation(rows*4,seed=rows);x[-2:]=0
    prepared=[prepare_mxfp4_moe(t,w) for t in x.chunk(4)]
    compact=[p.compact() for p in prepared]
    gathered=MoeInputCompact(*(torch.cat(ts) for ts in zip(*(p.tensors() for p in compact))))
    explicit=MoeInput(*(torch.cat(ts) for ts in zip(*(p.tensors() for p in prepared))))
    assert all(torch.equal(a,b) for a,b in zip(gathered.expand().tensors(),explicit.tensors()))
    expected=mxfp4_moe(explicit,w,return_partials=True)
    eager,replay=replay_check(lambda:list(mxfp4_moe(gathered,w,return_partials=True)))
    assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(expected,eager,replay))


def test_compact_validation(complete_weights):
    x=make_activation(8);p=prepare_mxfp4_moe(x,complete_weights);c=p.compact()
    with pytest.raises(ValueError):replace(c,correction_mask=c.correction_mask[:,:1]).expand()
    with pytest.raises(ValueError):replace(c,shared_scales=c.shared_scales.float()).expand()
    with pytest.raises(ValueError):replace(p,shared_codes=None,shared_scales=None).compact()
    with pytest.raises(ValueError):replace(p,routed_codes=p.routed_codes[:,::2]).compact()


@pytest.mark.parametrize('m',[1,3,17,513,8193])
def test_sidecar_logical_output_bounds(m):
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.launch import _run_compiled
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_pack import compile_sidecar_pack
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_unpack import compile_sidecar_unpack
    require_gfx950()
    x=make_activation(m,seed=900+m)
    q4,s4=mxfp4_quant(x);q8,s8=shared_quant(x)
    mask_guard=torch.full((m*768+32,),173,dtype=torch.uint8,device='cuda')
    out_guard=torch.full((m*3072+32,),173,dtype=torch.uint8,device='cuda')
    mask=mask_guard[16:-16];out=out_guard[16:-16]
    def call():
        args=(q8.view(-1),s8.view(-1),s4.view(-1))
        stream=torch.cuda.current_stream()
        _run_compiled(compile_sidecar_pack(),*args,q4.view(-1),mask,s4.numel(),stream)
        _run_compiled(compile_sidecar_unpack(),*args,out,mask,s4.numel(),stream)
    call();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    graph.replay();torch.cuda.synchronize()
    assert torch.equal(out,q4.view(-1))
    for guard in [mask_guard,out_guard]:
        assert (guard[:16]==173).all() and (guard[-16:]==173).all()


def test_sidecar_next_to_mfma():
    from tests.mfma_spinner import mfma_spinner
    require_gfx950()
    x=make_activation(8193,seed=9700)
    q4,s4=mxfp4_quant(x);q8,s8=shared_quant(x)
    expected=pack_mxfp4_sidecar(q4,s4,q8,s8)
    unpack_mxfp4_sidecar(expected,s4,q8,s8)
    torch.cuda.synchronize()
    spinner=mfma_spinner(256)
    stream=torch.cuda.Stream(priority=-1)
    stream.wait_stream(torch.cuda.current_stream())
    spinner(5_000_000,stream)
    outputs=[]
    for _ in range(3):
        mask=pack_mxfp4_sidecar(q4,s4,q8,s8)
        outputs.append((mask,unpack_mxfp4_sidecar(mask,s4,q8,s8)))
    torch.cuda.synchronize()
    assert all(torch.equal(mask,expected) and torch.equal(out,q4) for mask,out in outputs)
