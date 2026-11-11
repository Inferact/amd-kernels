# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Static peer embedding table, 100 changing ID batches, both capture orders.
Six mask rows and one anchor per request; no cold-table serving claim.
"""
import json,statistics as S,torch,torch.distributed as dist
from amd_kernels.embedding.vocab_gather_h6144 import VocabGather
from amd_kernels.embedding.vocab_gather_gemma_mxfp8_h6144 import vocab_gather_gemma_mxfp8
from amd_kernels.norm.gemma_rmsnorm_bf16_mxfp8_h6144 import gemma_rmsnorm_mxfp8
from tests.multi_gpu import run_on_4_gpus


def worker(rank,errors):
    torch.set_num_threads(8)
    dev=torch.device('cuda',rank)
    gen=torch.Generator(device=dev).manual_seed(913+rank)
    shard=51200
    weight=torch.randn((shard,6144),device=dev,dtype=torch.bfloat16,generator=gen)
    lookup=VocabGather(weight,rank,dist.group.WORLD)
    norm=torch.randn(6144,device=dev,dtype=torch.bfloat16,generator=gen)
    for requests in (1,10,16,32,64,128,256):
        m=requests*7
        # Every ID tensor is retained across both captures.
        inputs=[]
        for _ in range(100):
            ids=torch.full((requests,7),200058,device=dev,dtype=torch.int32)
            ids[:,0]=torch.randint(0,4*shard,(requests,),device=dev,generator=gen,dtype=torch.int32)
            inputs.append(ids.flatten())
        def fn(i,side):
            if side:return vocab_gather_gemma_mxfp8(inputs[i],lookup,norm)
            raw=lookup(inputs[i]);q,s=gemma_rmsnorm_mxfp8(raw,norm,1e-6,16)
            return q,s,raw
        fn(0,0);fn(0,1);torch.cuda.synchronize()
        for order in (0,1):
            graphs={}
            for side in ((0,1) if order==0 else (1,0)):
                gr=torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    for i in range(100):fn(i,side)
                graphs[side]=gr
            times=[[],[]]
            for r in range(5):
                for side in ((0,1) if (r+order)%2==0 else (1,0)):
                    graphs[side].replay();st,en=torch.cuda.Event(True),torch.cuda.Event(True);st.record()
                    for _ in range(10):graphs[side].replay()
                    en.record();en.synchronize();times[side].append(st.elapsed_time(en))
            every=[None]*4;dist.all_gather_object(every,times)
            if rank==0:
                vals=[[max(e[k][r] for e in every) for r in range(5)] for k in range(2)]
                print(json.dumps(dict(R=requests,order=order,old=vals[0],new=vals[1],unit='us')),flush=True)
    lookup.close()

if __name__=='__main__':run_on_4_gpus(worker,timeout=600)
