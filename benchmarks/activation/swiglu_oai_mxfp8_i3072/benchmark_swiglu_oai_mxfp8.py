# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``swiglu_oai_mxfp8`` (one launch) against the unfused chain: the aten
SwiGLU-OAI chain of the DSpark draft (7 elementwise kernels, bf16 after each) + the Triton MXFP8 quantizer
in the down GEMM's layout. One graph per side holds 100 calls on 100 distinct gate_up inputs (rows of the draft
batch: 7R for R = 1, 10, 16, 32, 64, 128, 256); median of 5 rounds of 10 replays, the two sides alternated
twice (the faster round printed), us per call.

    python3 benchmarks/activation/swiglu_oai_mxfp8_i3072/benchmark_swiglu_oai_mxfp8.py
"""

import statistics

import torch

from amd_kernels.activation.swiglu_oai_mxfp8_i3072 import swiglu_oai_mxfp8
from amd_kernels.activation.swiglu_oai_mxfp8_i3072.reference import swiglu_bf16_chain
from amd_kernels.gemm.mxfp8_common import gemm as G

I = 3072
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
ROWS = [7, 70, 112, 224, 448, 896, 1792]
CALLS = 100


def _time(graph, rounds=5, replays=10):
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(replays):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / (replays * CALLS) * 1e3)
    return statistics.median(out)


def main():
    for m in ROWS:
        tile = G.prequant_code_tile_rows(m, 6144, I)
        xs = [(torch.randn((m, 2 * I), device="cuda") * 3).bfloat16() for _ in range(CALLS)]

        def today(x):
            act = swiglu_bf16_chain(x, ALPHA, BETA, LIMIT)
            return G.mxfp8_e4m3_quantize_shuffled(act, tile) if tile else G.mxfp8_e4m3_quantize_aiter(act)

        def ours(x):
            return swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile)

        graphs = {}
        for name, fn in (("today", today), ("ours", ours)):
            fn(xs[0])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for x in xs:
                    fn(x)
            graphs[name] = g
        t = {k: [] for k in graphs}
        for _ in range(2):
            for k, g in graphs.items():
                t[k].append(_time(g))
        best = {k: min(v) for k, v in t.items()}
        print(f"rows={m:5d} tile={tile:3d}: aten chain + quantizer {best['today']:6.2f} us | "
              f"swiglu_oai_mxfp8 {best['ours']:6.2f} us | {best['today'] / best['ours']:.1f}x", flush=True)


if __name__ == "__main__":
    main()
