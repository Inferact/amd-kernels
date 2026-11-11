# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Paired BF16 SwiGLU epilogue: 100 graph inputs and 12 rotated weight copies (>400 MB)."""
import argparse
import json
import statistics
import torch
from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_gateup_swiglu_n6144_k6144_i3072 import host as GUS
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 7, 8])
    parser.add_argument('--rounds', type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1973)
    q, s = quantize_mxfp8((torch.randn((6144, 6144), device='cuda') * .02).bfloat16())
    w, ws = G.preshuffle_mxfp8_weight(q, s)
    weights = [(w.clone(), ws.clone()) for _ in range(12)]
    del q, s, w, ws
    for m in args.rows:
        assert 1 <= m <= 8
        inputs = [G.mxfp8_e4m3_quantize_shuffled(torch.randn((m, 6144), device='cuda', dtype=torch.bfloat16), 16) for _ in range(100)]
        GUS.warmup_mxfp8_gateup_swiglu_bf16(*inputs[0], *weights[0])
        for form in ('fp32', 'bf16_chain'):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for i, inp in enumerate(inputs):
                    GUS.mxfp8_gateup_swiglu_bf16(*inp, *weights[i % 12], m, form=form)
            samples = []
            for _ in range(args.rounds):
                graph.replay()
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(10):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))  # ms / 1000 calls == us / call
            print(json.dumps(dict(rows=m, form=form, us=statistics.median(samples), samples=samples)), flush=True)


if __name__ == '__main__':
    main()
