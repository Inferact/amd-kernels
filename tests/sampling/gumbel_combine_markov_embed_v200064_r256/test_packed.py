# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``gumbel_combine_markov_embed_packed`` against ``reference.py``, exact: owned rows 1-64, every step column, partials
with NaN blocks (first wins, also in the last block 195), ties of the maximum (first wins), -0.0 / +0.0 ties, all -inf
rows, vocab ids 0 / 50015 / 50016 / 200063; graph replay; no compile after the warm-up."""
import pytest
import torch
from amd_kernels.sampling import gumbel_combine_markov_embed_v200064_r256 as gc
from amd_kernels.sampling.gumbel_combine_markov_embed_v200064_r256.reference import (
    NB,V,RANK,gumbel_combine_markov_embed_reference,
)


def make_partials(n, gen):
    lm = torch.randn(n, NB, generator=gen) * 4 + 10
    la = torch.randint(0, V, (n, NB), generator=gen)
    for r in range(n):
        k = r % 8
        if k == 1:  # a NaN block wins over a larger value
            lm[r, 37] = float("nan")
            lm[r, 100] = 1e30
        elif k == 2:  # two NaN blocks: the first wins; one in the last block
            lm[r, [195, 60]] = float("nan")
        elif k == 3:  # a tie of the maximum: the first wins
            lm[r, [5, 77, 150]] = 1e6
        elif k == 4:  # -0.0 and +0.0 tie as the maximum
            lm[r] = -1.0
            lm[r, 9] = -0.0
            lm[r, 3] = 0.0
        elif k == 5:  # all -inf
            lm[r] = float("-inf")
        elif k == 6:  # the maximum in the last block and edge vocab ids
            lm[r, 195] = 1e7
            la[r, 195] = 200063
            la[r, 0] = 0
        elif k == 7:
            lm[r, 64] = 1e7
            la[r, 64] = (50015, 50016)[r % 2]
    return lm.cuda(), la.cuda()


@pytest.fixture(scope="module")
def w1():return torch.randn(V,RANK,device="cuda",dtype=torch.bfloat16)


def unpack(p,n):
    return p.view(-1,8,4,16,8).permute(0,3,1,2,4).contiguous().view(-1,256)[:n]


@pytest.mark.parametrize("n",[1,3,8,16,17,32,33,64])
@pytest.mark.parametrize("step",[0,3,6])
def test_packed_reference(w1,n,step):
    gen=torch.Generator().manual_seed(n*100+step)
    mx,ax=make_partials(n,gen)
    rows=torch.randperm(256,device="cuda")[:n]
    pad=16 if n<=16 else (32 if n<=32 else 64)
    p=torch.empty(pad,256,device="cuda",dtype=torch.bfloat16)
    ids=torch.full((256,7),-5,device="cuda",dtype=torch.int64)
    prev=torch.randint(V,(256,),device="cuda")
    ids_ref,prev_ref=ids.clone(),prev.clone()
    embed_ref=torch.zeros(256,256,device="cuda",dtype=torch.bfloat16)
    gumbel_combine_markov_embed_reference(mx,ax,rows,w1,ids_ref,step,prev_ref,embed_ref)
    gc.gumbel_combine_markov_embed_packed(mx,ax,rows,w1,ids,step,prev,p)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gc.gumbel_combine_markov_embed_packed(mx,ax,rows,w1,ids,step,prev,p)
    for _ in range(3):
        p.fill_(float("nan"));graph.replay();torch.cuda.synchronize()
        assert torch.equal(ids,ids_ref) and torch.equal(prev,prev_ref)
        assert torch.equal(unpack(p,n).view(torch.int16),embed_ref[rows].view(torch.int16))


def test_packed_no_compile_after_warmup(w1,monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler
    from amd_kernels.sampling.gumbel_combine_markov_embed_v200064_r256 import host
    host._packed_kernel_exe.cache_clear()
    gc.warmup_gumbel_combine_markov_embed(torch.device("cuda"))
    compiled=[];original=MlirCompiler.compile
    def track(*args,**kwargs):
        compiled.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(MlirCompiler,"compile",track)
    for n in (1,3,8,16,17,32,33,64):
        mx,ax=make_partials(n,torch.Generator().manual_seed(n))
        rows=torch.arange(n,device="cuda")
        p=torch.empty(16 if n<=16 else (32 if n<=32 else 64),256,device="cuda",dtype=torch.bfloat16)
        ids=torch.empty(n,7,device="cuda",dtype=torch.int64)
        prev=torch.empty(256,device="cuda",dtype=torch.int64)
        for step in range(7):
            gc.gumbel_combine_markov_embed_packed(mx,ax,rows,w1,ids,step,prev,p)
    torch.cuda.synchronize()
    assert compiled==[]
