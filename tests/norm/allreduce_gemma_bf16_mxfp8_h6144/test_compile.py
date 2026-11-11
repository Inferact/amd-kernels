# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Warm-up covers all row tiers, code layouts, output types and graph registration."""
import torch
import torch.distributed as dist
from amd_kernels.norm.allreduce_gemma_bf16_mxfp8_h6144 import Bf16ArNormComm
from tests.multi_gpu import run_on_4_gpus


def worker(rank,errors):
    from flydsl.compiler.jit_function import MlirCompiler
    dev=torch.device('cuda',rank)
    c=Bf16ArNormComm(rank,dist.group.WORLD,dev)
    w=torch.ones(6144,device=dev,dtype=torch.bfloat16)
    c.warm_up(w)
    orig=MlirCompiler.compile
    def unexpected(*a,**kw):raise AssertionError('compile after warm-up')
    MlirCompiler.compile=unexpected
    try:
        for m in (1,7,8,70,112,224,256,257,448,512,513,896,1024,1025,1792,2048):
            x=torch.zeros((m,6144),device=dev,dtype=torch.bfloat16)
            g=torch.cuda.CUDAGraph()
            for ctx in (None,g):
                def calls():
                    for owner in (False,True):
                        c.ar_add_rmsnorm(x,x,w,owned_residual=owner)
                        for tile in (0,16,32,64,128):c.ar_add_rmsnorm_mxfp8(x,x,w,1e-6,tile,owned_residual=owner)
                if ctx is None:calls()
                else:
                    with c.capture(),torch.cuda.graph(g):calls()
                    g.replay()
                torch.cuda.synchronize()
    finally:MlirCompiler.compile=orig
    c.close()


def test_warmup():run_on_4_gpus(worker,timeout=180)
