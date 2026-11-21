# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One GPU, cold rotated weights (6 copies of the 77 MB MLP), CUDA-graph replay, min / median us per call.

Stock: vLLM ``rocm_unquantized_gemm_impl`` (gate_up, down) and ``silu_and_mul``, the ops the drafter runs.
This entry: ``dense_mlp`` (two launches) on packed activations; roofline = the weights at the measured cold-read
ceiling (6.5 TB/s) plus two launch floors of 1.6 us.
Run: python3 benchmarks/decoder/bf16_dense_mlp_h7168_i1792/benchmark_mlp.py
"""
import argparse
import statistics

import torch

from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792 import MlpWorkspace, dense_mlp, host, pack_down, pack_gate_up, pack_rows
from amd_kernels.decoder.bf16_dense_mlp_h7168_i1792.reference import HIDDEN, INTER

WEIGHT_BYTES = (2 * INTER * HIDDEN + HIDDEN * INTER) * 2
CEIL_BW = 6.5e12
LAUNCH_US = 1.6


def timeit(fns, n, rounds, reps):
    graphs = []
    for fn in fns:
        for i in range(2):
            fn(i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(n):
                fn(i)
        graphs.append(g)
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    ts = [[] for _ in graphs]
    for r in range(rounds):
        for k in (range(len(graphs)) if r % 2 == 0 else range(len(graphs) - 1, -1, -1)):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(reps):
                graphs[k].replay()
            b.record()
            b.synchronize()
            ts[k].append(a.elapsed_time(b) * 1e3 / (reps * n))
    return ts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[32, 52, 60, 144])
    p.add_argument("--rounds", type=int, default=15)
    a = p.parse_args()
    from vllm.model_executor.layers.utils import rocm_unquantized_gemm_impl as gemm

    dev = torch.device("cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    nc = 6
    gu = [(torch.randn(2 * INTER, HIDDEN, device=dev, generator=g) * 0.02).bfloat16() for _ in range(nc)]
    dn = [(torch.randn(HIDDEN, INTER, device=dev, generator=g) * 0.03).bfloat16() for _ in range(nc)]
    gup = [pack_gate_up(w) for w in gu]
    dnp = [pack_down(w) for w in dn]
    ws = MlpWorkspace(dev)
    print("rows | stock us (min/med) | new us (min/med) | speedup | new floor us | % of floor", flush=True)
    for rows in a.rows:
        x = (torch.randn(rows, HIDDEN, device=dev, generator=g) * 0.5).bfloat16()
        xp = pack_rows(x, host.packed_tiles(rows))
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)

        def stock(i):
            y = gemm(x, gu[i % nc])
            act = torch.empty(rows, INTER, dtype=torch.bfloat16, device=dev)
            torch.ops._C.silu_and_mul(act, y)
            return gemm(act, dn[i % nc])

        def new(i):
            return dense_mlp(ws, xp, gup[i % nc], dnp[i % nc], rows, out=out)

        ts = timeit([stock, new], 24, a.rounds, 10)
        (s0, s1), (n0, n1) = [(min(t), statistics.median(t)) for t in ts]
        floor = WEIGHT_BYTES / CEIL_BW * 1e6 + 2 * LAUNCH_US
        print(f"{rows:4d} | {s0:6.1f} / {s1:6.1f} | {n0:6.1f} / {n1:6.1f} | {s0 / n0:4.2f}x | {floor:5.1f} | {floor / n0 * 100:3.0f}%",
              flush=True)


if __name__ == "__main__":
    main()
