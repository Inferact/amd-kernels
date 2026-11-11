# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SP payloads keep direct BF16 quantization and bypass full-M routing and quant."""
from dataclasses import replace
import importlib

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInput, mxfp4_moe, prepare_mxfp4_moe
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import mxfp4_moe_experts
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert import shared_expert
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
from .test_shared_expert import weights, complete_weights
from .workload import make_activation, make_routing, replay_check


@pytest.mark.parametrize('m', [1,8,256,512,513,1024,1025,2047,2048,2049,2689,4096,4097,
                              5377,8192,8193,10881,16384,16385,21761,24577,32768])
def test_prepared_quantization_and_replay(complete_weights,m):
    w=complete_weights;x=make_activation(m,seed=m);x[0,:64]=0
    expected=(*mxfp4_quant(x),*shared_quant(x))
    eager,replay=replay_check(lambda:list(prepare_mxfp4_moe(x,w).tensors()))
    assert all(torch.equal(a,b) for a,b in zip(eager,replay))
    q,s,wt,ids,q8,s8=replay
    assert all(torch.equal(a,b.view(torch.uint8)) for a,b in zip((q,s,q8,s8),expected))
    assert s8.shape==(m,192)  # no per-shard tiled padding in the wire payload
    refwt,refids=importlib.import_module(
        'amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain').route(x,w)
    assert torch.equal(ids,refids) and torch.equal(wt,refwt)


@pytest.mark.parametrize('m',[8,128,256,257,513,1025,2048,4096,8192,16384,32768])
def test_prepared_public_entry(complete_weights,m,monkeypatch):
    w=complete_weights
    x=make_activation(m,seed=m)
    prepared=prepare_mxfp4_moe(x,w)
    expected=list(mxfp4_moe(x,w,return_partials=True))
    # An already-gathered call must never revisit BF16 routing or quantization.
    def fail(*args,**kw):raise AssertionError('prepared call rerouted or requantized its input')
    monkeypatch.setattr('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain._prepare',fail)
    monkeypatch.setattr('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant.mxfp4_quant',fail)
    monkeypatch.setattr('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert.shared_quant',fail)
    def call():
        return list(mxfp4_moe(prepared,w,return_partials=True))
    eager,replay=replay_check(call)
    assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(expected,eager,replay))
    acc=torch.zeros_like(expected[0][:,0],dtype=torch.float32)
    for k in range(4):acc=acc+expected[0][:,k].float()
    acc=acc+expected[1].float()
    assert torch.equal(mxfp4_moe(prepared,w),acc.bfloat16())


@pytest.mark.parametrize('rows',[3,129,513,2049,8192])
def test_four_arbitrary_shards(complete_weights,rows):
    """Four shards gathered in rank order on one GPU (odd boundaries, padded tail tokens): the payload layout and its
    consumption, not IPC; the experts against the same routing, the routing against FP64."""
    w=complete_weights
    gen=torch.Generator(device='cuda').manual_seed(rows)
    gate=torch.randn((128,6144),device='cuda',generator=gen)/6144**.5
    w=replace(w,router_weight=gate,router_weight_lines=router_line_layout(gate))
    x=make_activation(rows*4,seed=rows);x[-2:]=0
    local=[prepare_mxfp4_moe(t,w) for t in x.chunk(4)]
    payload=MoeInput(*(torch.cat(ts) for ts in zip(*(p.tensors() for p in local))))
    payload.validate(w)
    q,s=mxfp4_quant(x);q8,s8=shared_quant(x)
    assert all(torch.equal(a,b.view(torch.uint8)) for a,b in zip(
        (payload.routed_codes,payload.routed_scales,payload.shared_codes,payload.shared_scales),(q,s,q8,s8)))
    scores=(x[:64].double()@gate.double().t()).sigmoid()
    ids=(scores+w.correction_bias.double()).topk(4).indices.int()
    assert torch.equal(ids,payload.topk_ids[:64])
    wt=scores.gather(1,ids.long());wt=(wt/wt.sum(1,keepdim=True)*2).float()
    torch.testing.assert_close(wt,payload.topk_weights[:64],atol=1e-6,rtol=1e-6)
    if rows*4<=512:  # the shared tiles folded into the small-tile launches
        from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_small import a4w4_experts
        expected,expected_shared=a4w4_experts((q,s),w.w13,w.w2,w.a4w4_scales,payload.topk_weights,payload.topk_ids,
                                              shared=(q8,s8,w.shared_weights))
    else:
        expected=mxfp4_moe_experts((q,s),payload.topk_weights,payload.topk_ids,w)
        expected_shared=shared_expert(x,w.shared_weights,x_quant=(q8,s8))
    def call():return list(mxfp4_moe(payload,w,return_partials=True))
    eager,replay=replay_check(call)
    assert torch.equal(expected,replay[0])
    assert not expected_shared[-2:].any() and not replay[1][-2:].any()
    cos=torch.nn.functional.cosine_similarity(expected_shared[:-2].float(),replay[1][:-2].float(),dim=1)
    assert cos.min()>=.9999
    assert all(torch.equal(a,b) for a,b in zip(eager,replay))


@pytest.mark.parametrize('routing',['uniform','hot','four'])
def test_prepared_changed_inputs_and_routing(complete_weights,routing):
    w=complete_weights;m=2052
    payload=prepare_mxfp4_moe(make_activation(m),w)
    mxfp4_moe(payload,w);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=mxfp4_moe(payload,w)
    previous=None
    for seed in [67,68,69]:
        updated=prepare_mxfp4_moe(make_activation(m,seed=seed),w)
        wt,ids=make_routing(m,seed=seed,skewed=routing=='hot')
        if routing=='four':ids[:]=torch.arange(4,device='cuda',dtype=torch.int32)
        updated=replace(updated,topk_weights=wt,topk_ids=ids)
        for dst,src in zip(payload.tensors(),updated.tensors()):dst.copy_(src)
        out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
        expected=mxfp4_moe(updated,w)
        assert torch.equal(out,expected) and torch.isfinite(out).all()
        if previous is not None:assert not torch.equal(out,previous)
        previous=out.clone()


@pytest.mark.parametrize('field,value',[('routed_codes',None),('routed_scales',None),
    ('shared_codes',None),('shared_scales',None),('topk_ids',None)])
def test_invalid_prepared_payload(complete_weights,field,value):
    p=prepare_mxfp4_moe(make_activation(8),complete_weights)
    bad=replace(p,**{field:value})
    with pytest.raises(ValueError):mxfp4_moe(bad,complete_weights)


@pytest.mark.parametrize('m',[1025,2049,4097,8193])
def test_row_major_dual_quant_guards(complete_weights,m,monkeypatch):
    import math
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import shared_mxfp8_quant_ref,mxfp4_quant_ref
    w=complete_weights;x=make_activation(m,seed=m)
    x[:16]*=torch.exp2(torch.linspace(-96,96,16,device=x.device))[:,None]
    x[0]=0
    reference=(*mxfp4_quant_ref(x),*shared_mxfp8_quant_ref(x))
    original=torch.empty;guards=[]
    def empty(*size,**kw):
        shape=tuple(size[0]) if isinstance(size[0],(tuple,list,torch.Size)) else tuple(size)
        dt=kw.get('dtype',torch.float32);g=128//torch.tensor([],dtype=dt).element_size()
        storage=original((math.prod(shape)+2*g,),**kw);storage.fill_(19)
        out=storage[g:-g].view(shape);out.fill_(float('nan') if dt.is_floating_point else 255)
        guards.append((storage,g));return out
    monkeypatch.setattr(torch,'empty',empty)
    p=prepare_mxfp4_moe(x,w);torch.cuda.synchronize()
    assert all(torch.equal(a,b) for a,b in zip(
        (p.routed_codes,p.routed_scales,p.shared_codes,p.shared_scales),reference))
    assert all((t[:g]==19).all() and (t[-g:]==19).all() for t,g in guards)
