# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Whole dense FFN at 1-16 rows, TP2 (gate_up N 12288 K 6144, down N 6144 K 6144): the fused gate_up + SwiGLU kernel +
the down linear against what vLLM runs at TP2 without it (Triton ``tl.dot_scaled`` gate_up on row-major codes, the
form's SwiGLU -- Triton ``swiglu_oai_split`` for ``fp32``, the aten chain for ``bf16_chain`` -- and the same down
linear). Graph replay of 100 calls over 6 rotated weight copies (> 900 MB per side), both sides alternated over 5
rounds, median us per FFN. ``--sweep`` times the fused kernel alone over K-wave counts and lookaheads (each checked
bit for bit against the decode GEMM of the same K-wave count first).

    python3 benchmarks/gemm/mxfp8_gateup_swiglu_n12288_k6144_i6144/benchmark_gateup_swiglu_bf16.py [--rows 1 7 8] [--sweep]
"""
import argparse
import statistics

import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import quantize_mxfp8, swiglu_bf16_reference
from amd_kernels.gemm.mxfp8_gateup_swiglu_n12288_k6144_i6144 import host as GUS

N, K, I = GUS.N, GUS.K, GUS.I
ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0
COPIES, CALLS = 6, 100
SWEEP = [(kw, (lw, la)) for kw in (1, 2, 3, 4) for lw in (4, 5, 6, 7, 8) for la in (4, 6, 7, 8)]
FINALISTS = [(1, (7, 7)), (1, (7, 8)), (1, (8, 6)), (2, (7, 7)), (2, (7, 8)), (3, (4, 8)), (3, (5, 8)), (3, (4, 7))]


def _time(graph, rounds):
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / (10 * CALLS) * 1e3)
    return statistics.median(samples)


def _capture(fn):
    fn(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(CALLS):
            fn(i)
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, nargs='+', default=[1, 2, 4, 7, 8, 14, 16])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--finalists', action='store_true', help='sweep only the shortlisted configs')
    args = parser.parse_args()
    torch.manual_seed(1973)
    gq, gs = quantize_mxfp8((torch.randn((N, K), device='cuda') * .02).bfloat16())
    dq, ds = quantize_mxfp8((torch.randn((K, I), device='cuda') * .02).bfloat16())
    gate_up = [G.preshuffle_mxfp8_weight(gq, gs) for _ in range(COPIES)]
    down = [G.preshuffle_mxfp8_weight(dq, ds) for _ in range(COPIES)]
    row_major = [(gq.clone(), gs.clone()) for _ in range(COPIES)]
    for m in args.rows:
        assert 1 <= m <= 16
        xs = [(torch.randn((m, K), device='cuda') * 2).bfloat16() for _ in range(CALLS)]
        tiled = [G.mxfp8_e4m3_quantize_shuffled(x, 16) for x in xs]
        if args.sweep or args.finalists:
            q, s = tiled[0]
            cells = []
            for config in FINALISTS if args.finalists else SWEEP:
                gu = torch.empty((m, N), device='cuda', dtype=torch.bfloat16)
                G._launch_decode(G._Config(16, 32, 128, 0, 0, 1, 'decode', config[0], 0, 7, 7), q, s, *gate_up[0], gu)
                forms = ('fp32',) if config[0] == 1 else ('fp32', 'bf16_chain')
                for form in forms:
                    got = GUS.mxfp8_gateup_swiglu_bf16(q, s, *gate_up[0], m, form=form, config=config)
                    if not torch.equal(got.view(torch.int16), swiglu_bf16_reference(gu, form=form).view(torch.int16)):
                        cells.append(f"{form} kw{config[0]} la{config[1]}: WRONG")
                        continue
                    us = _time(_capture(lambda i, f=form, c=config: GUS.mxfp8_gateup_swiglu_bf16(
                        *tiled[i], *gate_up[i % COPIES], m, form=f, config=c)), args.rounds)
                    cells.append(f"{form} kw{config[0]} la{config[1][0]}x{config[1][1]}: {us:.2f}")
            print(f"rows={m}: " + " | ".join(cells), flush=True)
            continue
        from vllm.model_executor.kernels.linear.mxfp8.rocm_native import _mxfp8_dot_scaled_gemm
        from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize
        from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_split
        rows = [mxfp8_e4m3_quantize(x) for x in xs]
        for form in ('fp32', 'bf16_chain'):
            act = (lambda h: swiglu_oai_split(h, ALPHA, BETA, LIMIT)) if form == 'fp32' else (
                lambda h: swiglu_bf16_reference(h, ALPHA, BETA, LIMIT, form))

            def baseline(i):
                h = _mxfp8_dot_scaled_gemm(*rows[i], *row_major[i % COPIES], torch.bfloat16)
                return G.mxfp8_linear(act(h), *down[i % COPIES])

            def fused(i):
                a = GUS.mxfp8_gateup_swiglu_bf16(*tiled[i], *gate_up[i % COPIES], m, ALPHA, BETA, LIMIT, form)
                return G.mxfp8_linear(a, *down[i % COPIES])

            def fused_only(i):
                return GUS.mxfp8_gateup_swiglu_bf16(*tiled[i], *gate_up[i % COPIES], m, ALPHA, BETA, LIMIT, form)

            graphs = {'baseline': _capture(baseline), 'fused': _capture(fused), 'kernel': _capture(fused_only)}
            t = {k: [] for k in graphs}
            for _ in range(2):
                for k, g in graphs.items():
                    t[k].append(_time(g, args.rounds))
            t = {k: min(v) for k, v in t.items()}
            print(f"rows={m} {form}: baseline FFN {t['baseline']:6.2f} us | fused FFN {t['fused']:6.2f} us | "
                  f"{t['baseline'] / t['fused']:.2f}x | fused kernel alone {t['kernel']:6.2f} us", flush=True)


if __name__ == '__main__':
    main()
