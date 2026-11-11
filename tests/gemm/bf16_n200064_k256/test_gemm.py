# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import pytest
import torch
from amd_kernels.gemm.bf16_n200064_k256 import (
    bf16_gemm_packed_rows,pack_bf16_rows,padded_rows,warmup_bf16_gemm_packed_rows,
)
from amd_kernels.gemm.bf16_n200064_k256 import host
from amd_kernels.gemm.bf16_n200064_k256.reference import (
    N,K,bf16_gemm_reference,pack_rows_reference,
)

@pytest.fixture(scope="module")
def weight():
    return torch.randn(N,K,device="cuda",dtype=torch.bfloat16)


def operands(n):
    x=torch.randn(256,K,device="cuda",dtype=torch.bfloat16)
    indices=torch.randperm(256,device="cuda")[:n]
    packed=torch.empty(padded_rows(n),K,device="cuda",dtype=torch.bfloat16)
    y=torch.full((n,N),float("nan"),device="cuda",dtype=torch.bfloat16)
    return x,indices,packed,y


def check_numeric(y,ref):
    yf,rf=y.float(),ref.float()
    cos=torch.nn.functional.cosine_similarity(yf,rf,dim=1)
    l2=(yf-rf).norm(dim=1)/rf.norm(dim=1)
    assert bool(torch.all(cos>=.99999)),cos.min().item()
    assert bool(torch.all(l2<=1e-3)),l2.max().item()
    assert (yf-rf).norm()/rf.norm()<=1e-3


@pytest.mark.parametrize("n",[0,1,2,3,4,8,15,16,17,24,31,32,33,48,63,64])
def test_reference(weight,n):
    x,idx,packed,y=operands(n)
    pack_bf16_rows(x,idx,packed)
    assert torch.equal(packed.view(torch.int16),pack_rows_reference(x,idx).view(torch.int16))
    bf16_gemm_packed_rows(packed,weight,n,y)
    ref=bf16_gemm_reference(packed,weight,n)
    if n:check_numeric(y,ref)
    old=y.clone()
    for _ in range(3):
        bf16_gemm_packed_rows(packed,weight,n,y)
        assert torch.equal(y.view(torch.int16),old.view(torch.int16))


def test_nonfinite_propagation(weight):
    w=weight.clone()
    w[0,2]=float("nan");w[50015,8]=float("inf");w[50016].zero_()
    x,idx,p,y=operands(3)
    idx.copy_(torch.arange(3,device="cuda"))
    x[0,1]=float("nan");x[1,3]=float("inf");x[2].zero_()
    pack_bf16_rows(x,idx,p);bf16_gemm_packed_rows(p,w,3,y)
    ref=bf16_gemm_reference(p,w,3)
    assert torch.equal(torch.isnan(y),torch.isnan(ref))
    assert torch.equal(torch.isposinf(y),torch.isposinf(ref))
    assert torch.equal(torch.isneginf(y),torch.isneginf(ref))
    finite=torch.isfinite(ref)
    assert torch.equal(y[finite],ref[finite])


def test_graph_no_compile_after_warmup(weight,monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler
    host._gemm.cache_clear();host._pack.cache_clear()
    warmup_bf16_gemm_packed_rows(torch.device("cuda"))
    compiled=[];original=MlirCompiler.compile
    def track(*args,**kwargs):
        compiled.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(MlirCompiler,"compile",track)
    for n in (1,3,16,17,31,32,33,48,64):
        x,idx,p,y=operands(n)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pack_bf16_rows(x,idx,p);bf16_gemm_packed_rows(p,weight,n,y)
        for _ in range(3):
            x.normal_();idx.copy_(torch.randperm(256,device="cuda")[:n]);y.fill_(float("nan"))
            graph.replay();torch.cuda.synchronize()
            check_numeric(y,bf16_gemm_reference(pack_rows_reference(x,idx),weight,n))
    assert compiled==[]


def test_next_to_mfma(weight):
    from tests.mfma_spinner import mfma_spinner
    spin=mfma_spinner(256);stream=torch.cuda.Stream(priority=-1)
    x,idx,p,y=operands(64);pack_bf16_rows(x,idx,p)
    bf16_gemm_packed_rows(p,weight,64,y);torch.cuda.synchronize();expected=y.clone()
    for _ in range(20):
        spin(100000,stream)
        bf16_gemm_packed_rows(p,weight,64,y)
        torch.cuda.synchronize()
        assert torch.equal(y.view(torch.int16),expected.view(torch.int16))
