# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired gate/up epilogue preserves materialized BF16 dots, both activations, and down input quantization."""
import pytest,torch
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_gateup_swiglu_n6144_k6144_i3072 import host as GUS
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8,swiglu_bf16_reference,mxfp8_gateup_swiglu_bf16_reference

def materialized_gu(q,s,pw,ps,m,form):
 # Match the served GEMM's K partition; activation/quantization remain bitwise.
 if form == 'bf16_chain':
  return G.mxfp8_gemm_shuffled(q,s,pw,ps,torch.bfloat16,m)
 cfg=G._Config(16,32,128,0,0,1,'decode',1,0,7,7)
 out=torch.empty((m,6144),device=q.device,dtype=torch.bfloat16)
 G._launch_decode(cfg,q,s,pw,ps,out)
 return out

@pytest.fixture(scope='module')
def weights():
 torch.manual_seed(1973)
 w=(torch.randn((6144,6144),device='cuda')*.02).bfloat16()
 q,s=quantize_mxfp8(w)
 return q,s,G.preshuffle_mxfp8_weight(q,s)

@pytest.mark.parametrize('form',('fp32','bf16_chain'))
@pytest.mark.parametrize('m',(1,3,7,8))
def test_fp64_and_graph(weights,form,m):
 w,s,(pw,ps)=weights
 torch.manual_seed(m)
 x=(torch.randn((m,6144),device='cuda')*torch.exp2(torch.randint(-5,6,(m,192,1),device='cuda').expand(m,192,32).reshape(m,6144).float())).bfloat16()
 if m>1:x[0].zero_()
 q,sc=G.mxfp8_e4m3_quantize_shuffled(x,16)
 got=GUS.mxfp8_gateup_swiglu_bf16(q,sc,pw,ps,m,form=form)
 gu=materialized_gu(q,sc,pw,ps,m,form)
 expected=swiglu_bf16_reference(gu,form=form)
 torch.cuda.synchronize()
 assert torch.equal(got.view(torch.int16),expected.view(torch.int16)),(m,form,int((got!=expected).sum()))
 qx,sx=quantize_mxfp8(x);ref=mxfp8_gateup_swiglu_bf16_reference(qx,sx,w,s,form=form)
 cos=torch.nn.functional.cosine_similarity(got.double(),ref.double(),dim=1)
 cos=torch.where(ref.double().norm(dim=1)==0,(got==0).all(dim=1).double(),cos)
 assert cos.min()>=.9999,(m,form,cos.min())
 gr=torch.cuda.CUDAGraph()
 with torch.cuda.graph(gr):out=GUS.mxfp8_gateup_swiglu_bf16(q,sc,pw,ps,m,form=form)
 for _ in range(3):
  out.fill_(float('nan'));gr.replay();torch.cuda.synchronize()
  assert torch.equal(out.view(torch.int16),got.view(torch.int16))
 q.view(torch.uint8).zero_();gr.replay();torch.cuda.synchronize();assert not out.count_nonzero()

@pytest.mark.parametrize('form',('fp32','bf16_chain'))
def test_many_fusion_inputs(weights,form):
 _,_,(pw,ps)=weights
 for seed in range(64):
  torch.manual_seed(seed+771)
  x=(torch.randn((8,6144),device='cuda')*2**(seed%11-5)).bfloat16()
  q,sc=G.mxfp8_e4m3_quantize_shuffled(x,16)
  gu=materialized_gu(q,sc,pw,ps,8,form)
  ref=swiglu_bf16_reference(gu,form=form)
  got=GUS.mxfp8_gateup_swiglu_bf16(q,sc,pw,ps,8,form=form)
  torch.cuda.synchronize()
  assert torch.equal(got.view(torch.int16),ref.view(torch.int16)),(form,seed,int((got!=ref).sum()))

def test_warmup(weights):
 from flydsl.compiler.jit_function import MlirCompiler
 _,_,(pw,ps)=weights
 x=torch.zeros((8,6144),device='cuda',dtype=torch.bfloat16)
 q,s=G.mxfp8_e4m3_quantize_shuffled(x,16)
 GUS.warmup_mxfp8_gateup_swiglu_bf16(q,s,pw,ps)
 orig=MlirCompiler.compile
 def unexpected(*a,**kw):raise AssertionError('compile after warm-up')
 MlirCompiler.compile=unexpected
 try:
  for m in range(1,9):
   for form in ('fp32','bf16_chain'):GUS.mxfp8_gateup_swiglu_bf16(q,s,pw,ps,m,form=form)
  torch.cuda.synchronize()
 finally:MlirCompiler.compile=orig


def test_mfma_co_residency(weights):
 from tests.mfma_spinner import mfma_spinner
 _,_,(pw,ps)=weights
 x=torch.randn((8,6144),device='cuda',dtype=torch.bfloat16)
 q,s=G.mxfp8_e4m3_quantize_shuffled(x,16)
 spinner=mfma_spinner(256);side=torch.cuda.Stream(priority=-1)
 for form in ('fp32','bf16_chain'):
  ref=GUS.mxfp8_gateup_swiglu_bf16(q,s,pw,ps,8,form=form)
  torch.cuda.synchronize()
  for _ in range(3):
   spinner(100_000,side)
   got=GUS.mxfp8_gateup_swiglu_bf16(q,s,pw,ps,8,form=form)
   torch.cuda.synchronize();assert torch.equal(ref.view(torch.int16),got.view(torch.int16))
