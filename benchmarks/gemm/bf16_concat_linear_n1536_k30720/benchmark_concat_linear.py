# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One GPU, cold weights: the N-shard GEMM + slab reduce (5 rotated packed 1536-column copies, 472 MB) vs
torch.cat + F.linear of the replicated weight (2 rotated copies, 755 MB), 20 distinct inputs per graph, 5
alternated rounds x 10 replays, us per call (median [min-max]). The 4-rank gather is not included."""
import argparse
import statistics
import torch
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720 import bf16_concat_linear, pack_weight

p = argparse.ArgumentParser()
p.add_argument('--rows', type=int, nargs='+', default=[8, 80, 128, 256, 512, 1024, 2048])
a = p.parse_args()
g = torch.Generator(device='cuda').manual_seed(0)
full = [(torch.randn((6144, 30720), device='cuda', generator=g) * 0.02).bfloat16() for _ in range(2)]
packed = [pack_weight(full[0][:1536].clone()) for _ in range(5)]
for m in a.rows:
    xs = [[torch.randn((m, 6144), device='cuda', generator=g).bfloat16() for _ in range(5)] for _ in range(20)]
    outs = [torch.empty((m, 1536), device='cuda', dtype=torch.bfloat16) for _ in range(20)]

    def old(i):
        return torch.nn.functional.linear(torch.cat(xs[i], 1), full[i % 2])

    def new(i):
        return bf16_concat_linear(xs[i], packed[i % 5], out=outs[i])
    for fn in (old, new):
        fn(0)
    torch.cuda.synchronize()
    graphs = []
    for fn in (old, new):
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            for i in range(20):
                fn(i)
        graphs.append(gr)
    for gr in graphs:
        gr.replay()
    torch.cuda.synchronize()
    ts = [[], []]
    for r in range(5):
        for k in ((0, 1) if r % 2 == 0 else (1, 0)):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(10):
                graphs[k].replay()
            e.record()
            e.synchronize()
            ts[k].append(s.elapsed_time(e) * 1e3 / 200)
    f = lambda v: '%.1f [%.1f-%.1f]' % (statistics.median(v), min(v), max(v))  # noqa: E731
    print(f'rows {m}: cat + F.linear (replicated) {f(ts[0])} us, N-shard GEMM + reduce {f(ts[1])} us', flush=True)
    del xs, graphs
