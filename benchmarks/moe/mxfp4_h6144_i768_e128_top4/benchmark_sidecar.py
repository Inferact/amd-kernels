# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Component cost of the optional lossless FP4 sidecar; no TP4 transport claim.

Inputs rotate beyond MALL. Quantization and routing are outside timing.
Packing replaces 3072 B/token FP4 codes by a 768 B/token correction mask.
A layer claim requires packing + gather + public entry + fused collective.
"""
import argparse
import json
import statistics

import numpy as np
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_pack import pack_mxfp4_sidecar
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.sidecar_unpack import unpack_mxfp4_sidecar


def measure(m,args):
    count=100 if m<=2048 else 16
    operands=[]
    for i in range(count):
        x=torch.from_numpy(np.random.default_rng(1000*m+i).standard_normal((m,6144),dtype=np.float32)).cuda().bfloat16()
        q4,s4=mxfp4_quant(x);q8,s8=shared_quant(x)
        mask=pack_mxfp4_sidecar(q4,s4,q8,s8)
        operands.append((q4,s4,q8,s8,mask))
    graphs={};outputs={}
    for name in ['pack','unpack']:
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name]=[pack_mxfp4_sidecar(q4,s4,q8,s8) if name=='pack'
                           else unpack_mxfp4_sidecar(mask,s4,q8,s8)
                           for q4,s4,q8,s8,mask in operands]
        graphs[name]=graph
    for _ in range(20):
        for graph in graphs.values():graph.replay()
        torch.cuda.synchronize()
    samples={name:[] for name in graphs}
    for round in range(args.rounds):
        for name in list(graphs) if round%2==0 else list(graphs)[::-1]:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):graphs[name].replay()
            end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/(args.replays*count))
    for name,graph in graphs.items():
        for t in outputs[name]:t.fill_(173)
        torch.cuda.synchronize();graph.replay();torch.cuda.synchronize()
        expected=[row[4 if name=='pack' else 0] for row in operands]
        assert all(torch.equal(t,e) for t,e in zip(outputs[name],expected)),name
    return dict(m=m,inputs=count,input_bytes=count*m*9600,
                explicit_bytes_per_token=9632,compact_bytes_per_token=7328,
                samples={name:dict(median=statistics.median(v),min=min(v),max=max(v),rounds=v)
                         for name,v in samples.items()})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokens',nargs='+',type=int,default=[1024,2048,4096,8192,16384,32768])
    ap.add_argument('--rounds',type=int,default=7)
    ap.add_argument('--replays',type=int,default=10)
    args=ap.parse_args()
    print(json.dumps(dict(scope='codec components only; no collective',host=__import__('os').uname().nodename,
                          pci=torch.cuda.get_device_properties(0).pci_bus_id)))
    for m in args.tokens:
        print(json.dumps(measure(m,args)),flush=True)
        torch.cuda.empty_cache()


if __name__=='__main__':main()
