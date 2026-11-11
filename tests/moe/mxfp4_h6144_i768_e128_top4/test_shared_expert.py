# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert import SharedWeights,shared_expert
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.reference import shared_mxfp8_ref,shared_mxfp8_quant_ref
from .workload import make_activation,require_gfx950,replay_check


@pytest.fixture(scope='module')
def weights():
    require_gfx950();g=torch.Generator(device='cuda').manual_seed(607)
    def weight(n,k):
        q=(torch.randn((n,k),device='cuda',generator=g)*20).to(torch.float8_e4m3fn).view(torch.uint8)
        s=torch.randint(118,124,(n,k//32),device='cuda',dtype=torch.uint8,generator=g)
        return q,s
    raw=(*weight(1536,6144),*weight(6144,768))
    return raw,SharedWeights.from_row_major(*raw)


@pytest.fixture
def separate_shared(monkeypatch):
    """Keep the standalone scheduling tests independent of the composed launch."""
    import importlib
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    es=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_bm128')
    monkeypatch.setattr(sh,'shared_in_persistent_for',lambda n:False)
    def fail(*args,**kwargs):raise AssertionError('standalone oracle called the composed kernel')
    monkeypatch.setattr(es,'_gemm12_shared',fail)


@pytest.mark.parametrize('m',[96,97,128,192,256,512,513,1024,2048,4096,8192,16384,32768])
def test_shared_contract(weights,m):
    raw,w=weights;x=make_activation(m,seed=m)
    eager,replay=replay_check(lambda:[shared_expert(x,w)])
    assert torch.equal(eager[0],replay[0]) and torch.isfinite(replay[0]).all()
    ref=shared_mxfp8_ref(x[:min(m,64)],*raw)
    cos=torch.nn.functional.cosine_similarity(replay[0][:len(ref)].float(),ref,dim=1)
    assert cos.min()>=.999,(m,cos.min().item())


def test_shared_quant_rule(weights):
    x=make_activation(8192);x[:3,:64]=0
    refq,refs=shared_mxfp8_quant_ref(x)
    for m in [1,8,512,8192]:
        q,s=shared_quant(x[:m])
        assert torch.equal(q,refq[:m]) and torch.equal(s,refs[:m])


@pytest.fixture(scope='module')
def complete_weights(weights):
    from .workload import make_experts
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights
    raw,w=weights
    routed,_=make_experts()
    gate=torch.zeros((128,6144),device='cuda');bias=torch.arange(128,device='cuda').float()/1280
    return MoeWeights.from_row_major(gate,bias,*routed,shared=w)


@pytest.mark.parametrize('m',[1,8,16,128,257,513,2048,4097,32768])
@pytest.mark.parametrize('parallel',[False,True])
def test_complete_entry(complete_weights,m,parallel,monkeypatch,separate_shared):
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    monkeypatch.setattr(sh,'shared_parallel_for',lambda n:parallel)
    w=complete_weights;x=make_activation(m,seed=m)
    eager,replay=replay_check(lambda:list(mxfp4_moe(x,w,return_partials=True)))
    assert all(torch.equal(a,b) for a,b in zip(eager,replay))
    p,s=replay
    assert p.shape==(m,4,6144) and s.shape==(m,6144)
    reduced=mxfp4_moe(x,w)
    expected=torch.zeros_like(s,dtype=torch.float32)
    for k in range(4):expected=expected+p[:,k].float()
    assert torch.equal(reduced,(expected+s.float()).bfloat16())


@pytest.mark.parametrize('m',[1,2,5,8,12,13,16,17,33,64,95,96,97,128,129,160,161,192,193,200,208,224,240,255,256,
                              257,258,300,320,383,384,385,416,447,448,449])
def test_shared_folded_into_routed_launches(weights,complete_weights,m):
    """Up to 448 rows the shared tiles run first in the routed 16 / 32-row GEMM1 / GEMM2 launches (from 449 the BM128
    launch): replay equal to eager, the shared output within the FP32 reference and, from 96 rows, cos >= 0.9999 to the
    standalone shared expert."""
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    raw,_=weights;w=complete_weights;x=make_activation(m,seed=300+m)
    assert sh.shared_folded_for(m)==(m<=448)
    eager,replay=replay_check(lambda:list(mxfp4_moe(x,w,return_partials=True)))
    assert all(torch.equal(a,b) for a,b in zip(eager,replay))
    p,s=replay
    assert torch.isfinite(p).all() and torch.isfinite(s).all()
    again=mxfp4_moe(x,w,return_partials=True)
    assert torch.equal(again[0],p) and torch.equal(again[1],s)
    ref=shared_mxfp8_ref(x[:min(m,64)],*raw)
    assert torch.nn.functional.cosine_similarity(s[:len(ref)].float(),ref,dim=1).min()>=.999
    if m>=96:  # the standalone shared expert (shared_dense) on the same operands
        s0=shared_expert(x,w.shared_weights,x_quant=shared_quant(x))
        cos=torch.nn.functional.cosine_similarity(s.float(),s0.float(),dim=1)
        assert cos.min()>=.9999,(m,cos.min().item())


@pytest.mark.parametrize('m',[1,17,96,97,160,161,256,257,384,448])
@pytest.mark.parametrize('tile',[(16,4,2),(16,2,2),(32,2,2),(32,4,2),(16,1,2),(32,1,2),(64,4,1),(64,2,1),(32,2,1)])
def test_shared_fold_tiles_agree(weights,complete_weights,m,tile,monkeypatch):
    """Every folded shared tile (rows, GEMM1 K-waves, prefetch) gives the routed slots of the separate
    launches bit for bit and the shared output within FP32 summation order of the served tile."""
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    es=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_small')
    if m<=15 and tile!=(16,4,2):
        pytest.skip('inline routing keeps 16-row 4-K-wave shared tiles')
    w=complete_weights;x=make_activation(m,seed=500+m)
    p0,s0=mxfp4_moe(x,w,return_partials=True)
    monkeypatch.setattr(es,'shared_fold_tile_for',lambda n:tile)
    eager,replay=replay_check(lambda:list(mxfp4_moe(x,w,return_partials=True)))
    assert all(torch.equal(a,b) for a,b in zip(eager,replay))
    p,s=replay
    assert torch.equal(p,p0) and torch.isfinite(s).all()
    cos=torch.nn.functional.cosine_similarity(s.float(),s0.float(),dim=1)
    assert cos.min()>=.9999,(m,tile,cos.min().item())


def test_shared_weight_bits(weights):
    raw,w=weights
    restored=(w.w13.permute(0,3,1,2,4).reshape(1536,6144),
              w.s13.permute(0,5,3,1,4,2).reshape(1536,192),
              w.w2.permute(0,3,1,2,4).reshape(6144,768),
              w.s2.permute(0,5,3,1,4,2).reshape(6144,24))
    assert all(torch.equal(a,b.view(torch.uint8)) for a,b in zip(restored,raw))
    assert sum(getattr(w,k).numel() for k in ['w13','s13','w2','s2'])==sum(x.numel() for x in raw)


@pytest.mark.parametrize('m',[1,8,64,65,128,512,1024])
def test_dual_input_quant(complete_weights,m):
    """The top-k with both input quants in one launch: the MXFP4 and MXFP8 quantizers' bits, the plain top-k's routing."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_gemm import router_gemm_partials
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk import splitk_biased_topk
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_topk_mxfp4_mxfp8_quant import splitk_biased_topk_mxfp4_mxfp8_quant
    w=complete_weights;x=make_activation(m,seed=m)
    parts=router_gemm_partials(x,w.router_weight)
    wt=torch.empty((m,4),dtype=torch.float32,device=x.device);ids=torch.empty((m,4),dtype=torch.int32,device=x.device)
    splitk_biased_topk(parts,w.correction_bias,wt,ids,2.0)
    w2=torch.empty_like(wt);i2=torch.empty_like(ids)
    a,b=splitk_biased_topk_mxfp4_mxfp8_quant(parts,w.correction_bias,w2,i2,2.0,x)
    assert all(torch.equal(x,y.view(torch.uint8)) for x,y in zip(a,mxfp4_quant(x)))
    assert all(torch.equal(x,y) for x,y in zip(b,shared_quant(x)))
    assert torch.equal(w2,wt) and torch.equal(i2,ids)
    eager,replay=replay_check(lambda:[mxfp4_moe(x,w)])
    assert torch.equal(eager[0],replay[0])


@pytest.mark.parametrize('m',[3,128,513,2048,8192,32768])
def test_shared_dense(weights,m,monkeypatch):
    import importlib
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    raw,w=weights;x=make_activation(m,seed=m)
    monkeypatch.setattr(sh,'shared_dense_for',lambda n:True)
    eager,replay=replay_check(lambda:[shared_expert(x,w)])
    assert torch.equal(eager[0],replay[0]) and torch.isfinite(replay[0]).all()
    ref=shared_mxfp8_ref(x[:min(m,64)],*raw)
    cos=torch.nn.functional.cosine_similarity(replay[0][:len(ref)].float(),ref,dim=1)
    assert cos.min()>=.999,(m,cos.min().item())


@pytest.mark.parametrize('m',[3,128,513,8192])
@pytest.mark.parametrize('g1_tiles',[(2,2),(2,4)])
def test_shared_dense_guards(weights,m,g1_tiles,monkeypatch):
    import importlib,math
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    monkeypatch.setattr(sh,'shared_dense_for',lambda n:True)
    monkeypatch.setattr(sh,'shared_dense_config_for',lambda n:(g1_tiles,(2,2)))
    _,w=weights;x=make_activation(m)
    shared_expert(x,w);torch.cuda.synchronize()
    original=torch.empty;guards=[]
    def allocate(*size,**kw):
        dims=tuple(size[0]) if isinstance(size[0],(tuple,list)) else size
        dtype=kw.get('dtype',torch.float32);g=128//torch.tensor([],dtype=dtype).element_size()
        storage=original((math.prod(dims)+2*g,),**kw);storage.fill_(19)
        out=storage[g:-g].view(dims)
        if dtype==torch.bfloat16:out.fill_(float('nan'))
        guards.append((storage,g));return out
    monkeypatch.setattr(torch,'empty',allocate)
    eager,replay=replay_check(lambda:[shared_expert(x,w)])
    assert torch.equal(eager[0],replay[0]) and torch.isfinite(replay[0]).all()
    assert all((t[:g]==19).all() and (t[-g:]==19).all() for t,g in guards)


def test_shared_across_tiles(weights,monkeypatch):
    import importlib
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    _,w=weights;x=make_activation(8192)
    monkeypatch.setattr(sh,'shared_dense_for',lambda n:True)
    small=shared_expert(x[:8],w)
    large=shared_expert(x,w)[:8]
    cos=torch.nn.functional.cosine_similarity(small.float(),large.float(),dim=1)
    assert cos.min()>=.9999


@pytest.mark.parametrize('m',[3,128,129,256,513,1024,8192])
@pytest.mark.parametrize('g1_tiles',[(2,2),(2,4)])
def test_shared_dense_compact(weights,m,g1_tiles,monkeypatch):
    import importlib
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    _,w=weights;x=make_activation(m,seed=m)
    monkeypatch.setattr(sh,'shared_dense_for',lambda n:True)
    monkeypatch.setattr(sh,'shared_dense_config_for',lambda n:((4,4),(4,4)))
    large=shared_expert(x,w)
    monkeypatch.setattr(sh,'shared_dense_config_for',lambda n:(g1_tiles,(2,2)))
    eager,replay=replay_check(lambda:[shared_expert(x,w)])
    assert torch.equal(eager[0],replay[0])
    cos=torch.nn.functional.cosine_similarity(large.float(),replay[0].float(),dim=1)
    assert cos.min()>=.9999


@pytest.mark.parametrize('m',[513,1024,2048])
def test_dual_dense_overlap(complete_weights,m,monkeypatch,separate_shared):
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    w=complete_weights;x=make_activation(m,seed=m)
    monkeypatch.setattr(sh,'shared_dense_for',lambda n:True)
    monkeypatch.setattr(sh,'shared_parallel_for',lambda n:False)
    monkeypatch.setattr(sh,'shared_dual_quant_for',lambda n:False)
    reference=mxfp4_moe(x,w)
    monkeypatch.setattr(sh,'shared_parallel_for',lambda n:True)
    monkeypatch.setattr(sh,'shared_dual_quant_for',lambda n:n<=1024)
    eager,replay=replay_check(lambda:[mxfp4_moe(x,w)])
    assert torch.equal(eager[0],replay[0]) and torch.equal(reference,eager[0])


@pytest.mark.parametrize('m',[128,513,8192])
def test_shared_mfma_corunner(weights,m):
    from tests.mfma_spinner import mfma_spinner
    _,w=weights;x=make_activation(m)
    ref=shared_expert(x,w).clone();torch.cuda.synchronize()
    spinner=mfma_spinner(256);stream=torch.cuda.Stream(priority=-1)
    spinner(5_000_000,stream)
    out=shared_expert(x,w);torch.cuda.synchronize()
    assert torch.equal(out,ref)

@pytest.mark.parametrize('m',[1025,1536,2048,2049,4096,4097,8192,8193,16384,16385,24577,32768])
@pytest.mark.parametrize('parallel',[False,True])
def test_router_dual_shared_entry(complete_weights,m,parallel,monkeypatch,separate_shared):
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    monkeypatch.setattr(sh,'shared_parallel_for',lambda n:parallel)
    monkeypatch.setattr(sh,'shared_dual_quant_for',lambda n:n<=1024)
    w=complete_weights;x=make_activation(m,seed=m)
    old=mxfp4_moe(x,w)
    monkeypatch.setattr(sh,'shared_dual_quant_for',lambda n:True)
    eager,replay=replay_check(lambda:[mxfp4_moe(x,w)])
    assert torch.equal(old,eager[0]) and torch.equal(eager[0],replay[0])

@pytest.mark.parametrize('m',[4096,8192,16384,32768])
def test_router_dual_graph_changed_inputs(complete_weights,m,monkeypatch):
    from dataclasses import replace
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import router_line_layout
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    gate=torch.randn((128,6144),device='cuda')/6144**.5
    w=replace(complete_weights,router_weight=gate,router_weight_lines=router_line_layout(gate))
    x=make_activation(m,seed=m)
    mxfp4_moe(x,w);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=mxfp4_moe(x,w)
    previous=None
    for seed in [619,620,621]:
        x.copy_(make_activation(m,seed=seed))
        if seed==621:x.zero_()
        out.fill_(float('nan'))
        graph.replay();torch.cuda.synchronize()
        with monkeypatch.context() as mp:
            mp.setattr(sh,'shared_dual_quant_for',lambda n:n<=1024)
            mp.setattr(sh,'shared_in_persistent_for',lambda n:False)
            expected=mxfp4_moe(x,w)
        torch.cuda.synchronize()
        assert torch.equal(out,expected)
        if previous is not None:assert not torch.equal(out,previous)
        previous=out.clone()

@pytest.mark.parametrize('m',[4032,4096])
def test_partial_mode_uses_separate_shared_prep(complete_weights,m,monkeypatch,separate_shared):
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    mod=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.router_mxfp4_mxfp8_quant_line_layout')
    original=mod.router_mxfp4_mxfp8_quant_line_layout;calls=[]
    def counted(*args,**kw):
        calls.append(1)
        return original(*args,**kw)
    monkeypatch.setattr(mod,'router_mxfp4_mxfp8_quant_line_layout',counted)
    x=make_activation(m,seed=m);w=complete_weights
    p,s=mxfp4_moe(x,w,return_partials=True)
    assert not calls
    out=mxfp4_moe(x,w)
    assert len(calls)==1
    ref=torch.zeros_like(s,dtype=torch.float32)
    for k in range(4):ref=ref+p[:,k].float()
    assert torch.equal(out,(ref+s.float()).bfloat16())


def test_shared_schedule_tiers():
    """Folded through 448 rows, second stream 96-30720 (when not folded), serial outside."""
    import importlib
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    es=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_small')
    assert [sh.shared_in_persistent_for(m) for m in (448,449,512,513,1024,1025,1536,2048,3583,3584,3840,4096,4097)]==[False,True,True,True,True,True,True,True,True,False,False,False,False]
    assert [sh.shared_folded_for(m) for m in (1, 256, 257, 448, 449)] == [True, True, True, True, False]
    assert [es.shared_fold_tile_for(m) for m in (1, 96, 97, 128, 129, 384, 385, 448)] == [
        (16, 4, 2), (16, 4, 2), (16, 2, 2), (16, 2, 2), (64, 4, 1), (64, 4, 1), (64, 2, 1), (64, 2, 1)]
    assert [sh.shared_parallel_for(m) for m in (1, 95, 96, 16384, 16385, 28672, 28673, 30720, 30721, 32768)] == [
        False, False, True, True, True, True, True, True, False, False]


@pytest.mark.parametrize('m',[16385,28673,30720,30721])
def test_parallel_tier_bit_equal_serial(complete_weights,m,monkeypatch):
    """The served schedule (second stream through 30720) gives the serial schedule's bits, under replay too."""
    import importlib
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import mxfp4_moe
    sh=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_expert')
    w=complete_weights;x=make_activation(m,seed=m)
    eager,replay=replay_check(lambda:list(mxfp4_moe(x,w,return_partials=True)))
    monkeypatch.setattr(sh,'shared_parallel_for',lambda n:False)
    serial=list(mxfp4_moe(x,w,return_partials=True))
    assert all(torch.equal(a,b) and torch.equal(a,c) for a,b,c in zip(eager,replay,serial))


@pytest.mark.parametrize('m',[1025,2047,2048,8192,32768])
def test_shared_gemm2_nt_bit_equal(weights,m,monkeypatch):
    """Non-temporal shared GEMM2 stores from 2048 rows give the default policy's bits, under replay too."""
    import importlib
    sd=importlib.import_module('amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_dense')
    assert sd._gemm2_nt_for(m)==(m>=2048)
    assert sd._gemm2((4,4),nt=True) is not sd._gemm2((4,4))
    _,w=weights;x=make_activation(m,seed=m)
    eager,replay=replay_check(lambda:[shared_expert(x,w)])
    monkeypatch.setattr(sd,'_gemm2_nt_for',lambda n:n<sd.SHARED_GEMM2_NT_FROM)
    flipped=shared_expert(x,w)
    assert torch.equal(eager[0],replay[0]) and torch.equal(eager[0],flipped)
