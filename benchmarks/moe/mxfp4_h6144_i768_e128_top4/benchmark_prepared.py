# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Local SP preparation from 1025 rows: fused router/FP4/FP8 versus the fused router/FP4 and a separate FP8
quantization (below 1025 rows both quants share the top-k launch, with no separate counterpart).

Synthetic weights and rotating BF16 inputs; no communication or expert timing.
Both sides produce identical row-major operands/routing for the public entry.
"""
import argparse
import json
import statistics

import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeInput, prepare_mxfp4_moe
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.chain import _prepare
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
from benchmark_shared_moe import weights
from tests.moe.mxfp4_h6144_i768_e128_top4.workload import make_activation


def measure(w, n, args):
    xs = [make_activation(n, seed=1000*n+i) for i in range(100 if n <= 2048 else 16)]

    def separate(x):
        (q, s), wt, ids = _prepare(x, w)
        q8, s8 = shared_quant(x)
        return MoeInput(q, s.view(torch.uint8), wt, ids, q8, s8)

    arms = {'separate': separate, 'fused': lambda x: prepare_mxfp4_moe(x, w)}
    order = list(arms)[::-1] if args.reverse else list(arms)
    for x in xs:
        assert all(torch.equal(a, b) for a, b in zip(separate(x).tensors(), arms['fused'](x).tensors()))
    graphs, outputs = {}, {}
    for name in order:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = [arms[name](x).tensors() for x in xs]
        graphs[name] = graph
    for _ in range(20):
        for name in order:
            graphs[name].replay()
        torch.cuda.synchronize()
    samples = {name: [] for name in order}
    for i in range(args.rounds):
        for name in order if i % 2 == 0 else order[::-1]:
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(10):
                graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/(10*len(xs)))
    for a, b in zip(outputs['separate'], outputs['fused']):
        assert all(torch.equal(t, u) for t, u in zip(a, b))
    return {'local_tokens': n, 'scope': 'local prep only, no collective or experts',
            'times': {name: {'median_us': statistics.median(t), 'range_us': [min(t), max(t)], 'rounds_us': t}
                      for name, t in samples.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', nargs='+', type=int, default=[1025, 2048, 4096, 8192])
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    w = weights()
    for n in args.tokens:
        print(json.dumps(measure(w, n, args)), flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
