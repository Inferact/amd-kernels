# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Fused Gemma norm/quant graph latency; 100 distinct bf16 inputs per graph."""
import statistics, torch
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8

def main():
    for m in (7,8,70,80,112,128,224,256,448,512,896,1024,1792,2048):
        w=torch.randn(6144,device='cuda',dtype=torch.bfloat16)
        xs=[torch.randn((m,6144),device='cuda',dtype=torch.bfloat16) for _ in range(100)]
        for tile in (0,16):
            gemma_rmsnorm_mxfp8(xs[0],w,1e-6,tile);torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for x in xs:gemma_rmsnorm_mxfp8(x,w,1e-6,tile)
            ts=[]
            for _ in range(5):
                a,b=torch.cuda.Event(True),torch.cuda.Event(True);a.record()
                for _ in range(10):g.replay()
                b.record();b.synchronize();ts.append(a.elapsed_time(b))
            print(m,tile,statistics.median(ts),min(ts),max(ts),'us',flush=True)
if __name__=='__main__':main()
