# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The 65-80-row gate/up tile: same K order, 48-row producer padding, unchanged neighbouring rows."""
import dataclasses
import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8,mxfp8_gemm_reference


def test_dispatch_edges():
    for m in (65,69,70,79,80):
        cfg=G._select_config(m,6144,6144)
        assert (cfg.kernel,cfg.tile_m,cfg.tile_n,cfg.waves_m,cfg.split_k)==('decode',48,48,3,1)
        assert G.prequant_code_tile_rows(m,6144,6144)==48
    for m,expected in ((64,(64,32,3)),(81,(96,32,3)),(96,(96,32,3)),(97,(64,48,3))):
        cfg=G._select_config(m,6144,6144)
        assert (cfg.tile_m,cfg.tile_n,cfg.waves_m)==expected


@pytest.fixture(scope='module')
def weights():
    gen=torch.Generator(device='cuda').manual_seed(777)
    w=torch.randn((6144,6144),generator=gen,device='cuda')*.02
    w*=torch.exp2(torch.randint(-5,6,(6144,192,1),generator=gen,device='cuda').expand(6144,192,32).reshape(6144,6144).float())
    q,s=quantize_mxfp8(w.bfloat16())
    return q,s,G.preshuffle_mxfp8_weight(q,s)


@pytest.mark.parametrize('m',(65,69,70,79,80))
def test_old_order_and_fp64(weights,m):
    qw,sw,(w,ws)=weights
    gen=torch.Generator(device='cuda').manual_seed(m)
    x=torch.randn((m,6144),generator=gen,device='cuda')
    x*=torch.exp2(torch.randint(-5,6,(m,192,1),generator=gen,device='cuda').expand(m,192,32).reshape(m,6144).float())
    x=x.bfloat16();x[0].zero_()
    q,s=G.mxfp8_e4m3_quantize_shuffled(x,48)
    cfg=G._select_config(m,6144,6144)
    old=dataclasses.replace(cfg,tile_m=96,tile_n=32)
    out=torch.empty((m,6144),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(out)
    G._launch_decode(old,q,s,w,ws,ref)
    G._launch_decode(cfg,q,s,w,ws,out)
    torch.cuda.synchronize()
    assert torch.equal(out,ref)
    qx,sx=quantize_mxfp8(x)
    exact=mxfp8_gemm_reference(qx,sx,qw,sw)
    cos=torch.nn.functional.cosine_similarity(out[1:].double(),exact[1:].double(),dim=1)
    rel=(out.double()-exact.double()).norm()/exact.double().norm()
    assert cos.min()>=.99999 and rel<=1e-3,(cos.min(),rel)
    per_row_rel=(out[1:].double()-exact[1:].double()).norm(dim=1)/exact[1:].double().norm(dim=1)
    assert per_row_rel.max()<=1e-3,per_row_rel.max()
    gr=torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):G._launch_decode(cfg,q,s,w,ws,out)
    out.fill_(float('nan'));gr.replay();torch.cuda.synchronize()
    assert torch.equal(out,ref)


def test_warmup(weights):
    from flydsl.compiler.jit_function import MlirCompiler
    _,_,(w,ws)=weights
    x=torch.randn((80,6144),device='cuda',dtype=torch.bfloat16)
    q,s=G.mxfp8_e4m3_quantize_shuffled(x,48)
    out=torch.empty((80,6144),device='cuda',dtype=torch.bfloat16)
    cfg=G._select_config(80,6144,6144)
    G._launch_decode(cfg,q,s,w,ws,out);torch.cuda.synchronize()
    orig=MlirCompiler.compile
    def unexpected(*a,**kw):raise AssertionError('compile after warm-up')
    MlirCompiler.compile=unexpected
    try:
        for m in (65,69,70,79,80):
            o=torch.empty((m,6144),device='cuda',dtype=torch.bfloat16)
            G._launch_decode(G._select_config(m,6144,6144),q,s,w,ws,o)
        torch.cuda.synchronize()
    finally:MlirCompiler.compile=orig


@pytest.mark.parametrize('m', (70, 80))
def test_mfma_co_residency(weights, m):
    from tests.mfma_spinner import mfma_spinner
    _, _, (w, ws) = weights
    x = torch.randn((m, 6144), device='cuda', dtype=torch.bfloat16)
    tile = G.prequant_code_tile_rows(m, 6144, 6144)
    q, sc = G.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else G.mxfp8_e4m3_quantize_aiter(x)
    def call():
        if tile:
            return G.mxfp8_gemm_shuffled(q, sc, w, ws, torch.bfloat16, m)
        return G.mxfp8_gemm(q[:m], sc, w, ws, torch.bfloat16, activation_scale_preshuffled=True)
    ref = call()
    torch.cuda.synchronize()
    spinner = mfma_spinner(256)
    side = torch.cuda.Stream(priority=-1)
    for _ in range(3):
        spinner(100_000, side)
        got = call()
        torch.cuda.synchronize()
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
