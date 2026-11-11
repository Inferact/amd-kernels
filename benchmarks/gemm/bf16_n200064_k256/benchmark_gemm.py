# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the public projection: 100 activation sets, 3 timing copies
of the weight rotated once per seven calls. Model code keeps one weight copy.
"""
import statistics
import torch
from amd_kernels.gemm.bf16_n200064_k256 import (
    bf16_gemm_packed_rows,pack_bf16_rows,padded_rows,warmup_bf16_gemm_packed_rows,
)
from amd_kernels.gemm.bf16_n200064_k256.reference import N,K


def timing(calls):
    for f in calls:f()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:f()
    times=[]
    for _ in range(5):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):graph.replay()
        b.record();b.synchronize()
        times.append(a.elapsed_time(b)*1000/(10*len(calls)))
    return f"{statistics.median(times):.2f} [{min(times):.2f}-{max(times):.2f}]"


if __name__=="__main__":
    torch.manual_seed(441)
    dev=torch.device("cuda")
    warmup_bf16_gemm_packed_rows(dev)
    weight=torch.randn(N,K,dtype=torch.bfloat16,device=dev)
    weights=[weight,weight.clone(),weight.clone()]
    for n in (3,4,8,16,32,64):
        idx=torch.arange(n,device=dev)
        ps=[torch.empty(padded_rows(n),K,dtype=torch.bfloat16,device=dev) for _ in range(100)]
        for p in ps:pack_bf16_rows(torch.randn(n,K,dtype=torch.bfloat16,device=dev),idx,p)
        out=torch.empty(n,N,dtype=torch.bfloat16,device=dev)
        calls=[lambda p=p,w=weights[(i//7)%3]:bf16_gemm_packed_rows(p,w,n,out) for i,p in enumerate(ps)]
        print(f"rows={n} projection us={timing(calls)}",flush=True)
