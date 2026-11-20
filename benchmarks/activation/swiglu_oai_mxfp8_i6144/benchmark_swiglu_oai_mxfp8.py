# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``swiglu_oai_mxfp8`` (one launch) against what vLLM runs at TP2 without it: the draft's
aten SwiGLU-OAI chain (7 elementwise kernels, bf16 after each) or the target's Triton ``swiglu_oai_split``, each
followed by the MXFP8 quantizer in the down GEMM's layout. One graph per side holds 100 calls on 100 distinct
gate_up inputs (2.5 GB at 1024 rows, beyond the MALL); median of 5 rounds of 10 replays, the sides alternated
twice (the faster round printed), us per call. ``--sweep`` times every (pieces, threads) workgroup shape instead.

    python3 benchmarks/activation/swiglu_oai_mxfp8_i6144/benchmark_swiglu_oai_mxfp8.py [--form fp32] [--sweep]
"""

import argparse
import statistics

import torch

from amd_kernels.activation.swiglu_oai_mxfp8_i6144 import swiglu_oai_mxfp8
from amd_kernels.activation.swiglu_oai_mxfp8_i6144.reference import swiglu_bf16_chain
from amd_kernels.gemm.mxfp8_common import gemm as G

I = 6144
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
ROWS = [7, 8, 16, 56, 64, 112, 128, 224, 256, 448, 512, 896, 1024, 1792]
SHAPES = [(1, 768), (2, 384), (3, 256), (4, 192), (6, 128), (12, 64), (1, 384), (2, 192), (3, 128), (6, 64), (1, 256), (4, 64), (1, 192)]
CALLS = 100
HBM = 6.5e12


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


def _graphs(fns, xs):
    graphs = {}
    for name, fn in fns.items():
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
    return {k: min(v) for k, v in t.items()}


def _baseline(form, tile):
    def quant(act):
        return G.mxfp8_e4m3_quantize_shuffled(act, tile) if tile else G.mxfp8_e4m3_quantize_aiter(act)

    if form == "bf16_chain":
        return lambda x: quant(swiglu_bf16_chain(x, ALPHA, BETA, LIMIT))
    from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_split

    return lambda x: quant(swiglu_oai_split(x, ALPHA, BETA, LIMIT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="*", default=ROWS)
    ap.add_argument("--form", default="bf16_chain", choices=("bf16_chain", "fp32"))
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    for m in args.rows:
        tile = G.prequant_code_tile_rows(m, 6144, I)
        xs = [(torch.randn((m, 2 * I), device="cuda") * 3).bfloat16() for _ in range(CALLS)]
        floor = m * (2 * I * 2 + I + I // 32) / HBM * 1e6
        if args.sweep:
            fns = {s: (lambda x, s=s: swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile, args.form, shape=s)) for s in SHAPES}
            t = _graphs(fns, xs)
            best = min(t, key=t.get)
            cells = " ".join(f"{p}x{n}:{v:5.2f}" for (p, n), v in t.items())
            print(f"rows={m:5d} tile={tile:3d} {cells} | best {best}", flush=True)
            continue
        t = _graphs({"baseline": _baseline(args.form, tile),
                     "ours": lambda x: swiglu_oai_mxfp8(x, ALPHA, BETA, LIMIT, tile, args.form)}, xs)
        print(f"rows={m:5d} tile={tile:3d} {args.form}: baseline {t['baseline']:6.2f} us | swiglu_oai_mxfp8 "
              f"{t['ours']:6.2f} us | {t['baseline'] / t['ours']:.1f}x | HBM floor {floor:5.2f} us "
              f"({floor / t['ours']:.0%})", flush=True)


if __name__ == "__main__":
    main()
