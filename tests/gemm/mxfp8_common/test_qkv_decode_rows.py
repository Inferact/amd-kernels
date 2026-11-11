# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Owned N2304 schedules keep K order and all neighbouring rows; synthetic MXFP8 weights, spread scales."""
import dataclasses,pytest,torch
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8,mxfp8_gemm_reference

def test_edges():
 for m in (385,416,447,448):
  c=G._select_config(m,2304,6144);assert (c.kernel,c.tile_m,c.tile_n,c.waves_m,c.split_k)==('decode',64,64,3,1)
 for m in (769,800,896):
  c=G._select_config(m,2304,6144);assert (c.kernel,c.tile_m,c.tile_n,c.split_k)==('aiter',32,128,1)
 for m in (768,897,1024):
  c=G._select_config(m,2304,6144);assert (c.kernel,c.tile_m,c.tile_n)==('aiter',128,128)
 for m in (449,512):
  c=G._select_config(m,2304,6144);assert (c.kernel,c.tile_m,c.tile_n)==('decode',128,48)
 for m in (385,448,769,896):
  c=G._select_config(m,2304,6144,torch.float16);assert c.kernel=='aiter'

@pytest.fixture(scope='module')
def weights():
 torch.manual_seed(991)
 raw=torch.randn((2304,6144),device='cuda')*.02
 raw*=torch.exp2(torch.randint(-5,6,(2304,192,1),device='cuda').expand(2304,192,32).reshape(2304,6144).float())
 w,s=quantize_mxfp8(raw.bfloat16())

 return w,s,G.preshuffle_mxfp8_weight(w,s)

@pytest.mark.parametrize('m',(385,416,448,769,800,896))
def test_order_fp64_and_graph(weights,m):
 w,s,(pw,ps)=weights
 torch.manual_seed(m)
 x=torch.randn((m,6144),device='cuda')*torch.exp2(torch.randint(-5,6,(m,192,1),device='cuda').expand(m,192,32).reshape(m,6144).float())
 x=x.bfloat16();x[0].zero_();c=G._select_config(m,2304,6144)
 old=dataclasses.replace(c,tile_m=128,tile_n=48) if m<=448 else G._Config(128,128,256,2,4,1,'aiter')
 q,sc=G.mxfp8_e4m3_quantize_shuffled(x,128) if m<=448 else G.mxfp8_e4m3_quantize_aiter(x)
 out=torch.empty((m,2304),device='cuda',dtype=torch.bfloat16);ref=torch.empty_like(out)
 def launch(cfg,dst):
  if m<=448:G._launch_decode(cfg,q,sc,pw,ps,dst)
  else:G._launch_tiled(cfg,q,sc,pw,ps,dst)
 launch(old,ref);launch(c,out);torch.cuda.synchronize();assert torch.equal(out,ref)
 qx,sx=quantize_mxfp8(x);exact=mxfp8_gemm_reference(qx,sx,w,s,accumulate=torch.float64)
 diff=out[1:].double()-exact[1:].double()
 cos=torch.nn.functional.cosine_similarity(out[1:].double(),exact[1:].double(),dim=1)
 rel=diff.norm(dim=1)/exact[1:].double().norm(dim=1)
 assert cos.min()>=.99999 and rel.max()<=1e-3,(m,cos.min(),rel.max())
 gr=torch.cuda.CUDAGraph()
 with torch.cuda.graph(gr):launch(c,out)
 out.fill_(float('nan'));gr.replay();torch.cuda.synchronize();assert torch.equal(out,ref)


def test_compile_after_warmup(weights):
 from flydsl.compiler.jit_function import MlirCompiler
 _,_,(pw,ps)=weights
 def launch(m):
  x=torch.zeros((m,6144),device='cuda',dtype=torch.bfloat16)
  tile=G.prequant_code_tile_rows(m,2304,6144)
  q,sc=G.mxfp8_e4m3_quantize_shuffled(x,tile) if tile else G.mxfp8_e4m3_quantize_aiter(x)
  if tile:return G.mxfp8_gemm_shuffled(q,sc,pw,ps,torch.bfloat16,m)
  return G.mxfp8_gemm(q,sc,pw,ps,torch.bfloat16,activation_scale_preshuffled=True)
 launch(385);launch(769);torch.cuda.synchronize()
 orig=MlirCompiler.compile
 def unexpected(*a,**kw):raise AssertionError('compile after warm-up')
 MlirCompiler.compile=unexpected
 try:
  for m in (386,400,416,447,448,770,800,832,895,896):launch(m)
  torch.cuda.synchronize()
 finally:MlirCompiler.compile=orig


@pytest.mark.parametrize('m', (448, 896))
def test_mfma_co_residency(weights, m):
    from tests.mfma_spinner import mfma_spinner
    _, _, (w, ws) = weights
    x = torch.randn((m, 6144), device='cuda', dtype=torch.bfloat16)
    tile = G.prequant_code_tile_rows(m, 2304, 6144)
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
