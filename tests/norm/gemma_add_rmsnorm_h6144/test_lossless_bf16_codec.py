# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The 12-bit row code is exact: every 16-bit pattern, shuffled patterns, sparse exceptions and zeros (no float math:
NaN payloads and signed zeros are bytes too)."""
import os,sys,json
import torch
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Vector as Vec
from amd_kernels.common import buffer_ops
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.norm.gemma_add_rmsnorm_h6144.allreduce_add_rmsnorm.lossless_bf16 import enc_store,fetch,finish

@flyc.kernel(name='test_slots4_lossless12_roundtrip',known_block_size=[64,1,1])
def kernel(inp:fx.Int64,tmp:fx.Int64,raw:fx.Int64,out:fx.Int64):
    tx,b=fx.thread_idx.x,fx.block_idx.x
    rs=buffer_ops.create_buffer_resource_from_addr(inp+fx.Int64(b)*1024,num_records_bytes=1024)
    rd=Vec(buffer_ops.buffer_load(rs,tx*4,vec_width=4,dtype=fx.Int32))
    tp=tmp+fx.Int64(b)*12288;rp=raw+fx.Int64(b)*12288
    enc_store(rd,tp,rp,tx,fx.Boolean(True))
    fx.rocdl.s_waitcnt(vmcnt=0)
    fx.gpu.barrier()
    data=fetch(tp,tx,fx.Boolean(True))
    got=finish(data,rp,tx,fx.Boolean(True))
    dst=buffer_ops.create_buffer_resource_from_addr(out+fx.Int64(b)*1024,num_records_bytes=1024)
    buffer_ops.buffer_store(got,dst,tx*4)

@flyc.jit
def launch_slots4_lossless_roundtrip(inp:fx.Int64,tmp:fx.Int64,raw:fx.Int64,out:fx.Int64,n:fx.Int32,stream:fx.Stream):
    kernel(inp,tmp,raw,out).launch(grid=(n,1,1),block=(64,1,1),stream=stream)

def test_lossless_bf16_all_encodings():
    dev='cuda';torch.manual_seed(173)
    a=torch.arange(65536,dtype=torch.int32,device=dev).to(torch.int16)
    g=torch.Generator(device=dev).manual_seed(337)
    cases=[a,a[torch.randperm(65536,generator=g,device=dev)]]
    for k in range(4):
        t=torch.randn((128,512),generator=g,device=dev).bfloat16()
        if k==0:t[:,0]=2**-40  # one exception per wave
        if k==1:t[:,:2]=2**-40 # two exceptions per lane forces raw
        if k==2:t[:,::32]=0;t[:,1::32]=-0.0
        if k==3:
            t[:,:]=2**-128;t[:,::32]=2**-133;t[:,1::32]=-0.0
        cases.append(t.view(torch.int16).flatten())
    tmp=torch.empty((128,6144),dtype=torch.int16,device=dev);raw=torch.empty_like(tmp)
    out=torch.empty_like(a)
    for i,x in enumerate(cases):
        _run_compiled(launch_slots4_lossless_roundtrip,x.data_ptr(),tmp.data_ptr(),raw.data_ptr(),out.data_ptr(),128,torch.cuda.current_stream())
        assert torch.equal(x,out),(i,int((x!=out).sum()))
    print('codec round-trip: 6 x 65536 bf16 encodings, all bytes equal',flush=True)
