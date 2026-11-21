# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routed experts of one TP8 rank (E 896, top-16, H 3584, I 384): CUDA graph replay over 3 layers of weights (5.9 GB >> LLC),
interleaved min-of-N per layer, roofline vs the measured HBM ceiling and speedup vs the stock sequence.

Columns per (M, routing): unique experts, HBM bytes, then us / % of the measured ceiling for: this entry (experts only,
sort given) + the activation quant, the stock sequence vLLM runs (aiter ``fused_moe`` incl. its sort and quant, shipped
tuned rows) and the stock stage kernels alone (BM16 sort handed in, patched rows if AITER_CONFIG_FMOE points at them).

    python3 benchmarks/moe/mxfp4_h3584_i3072_e896_top16/benchmark_experts.py [M ...] [--dists uniform zipf1.0 cluster]
"""
import argparse
import os
import statistics

os.environ.setdefault("AITER_SITUV2_A4W4", "1")
import torch

from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16 import MoeWeights, Sort, mxfp4_moe_experts, quantize_activations
from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16.host import sort_capacity

E, H, I, TOPK = 896, 3584, 384, 16
HBM_CEILING = 7.0e12  # measured nontemporal HBM read on MI355X; 6.4-6.7 plain loads
EXPERT_BYTES = 2 * I * (H // 2) + 2 * I * (H // 32) + H * (I // 2) + H * 16  # weights + shuffled scales (2,193,408 B)
SERVED_M = [12, 24, 40, 65, 80, 128, 192]


def routing(m, dist, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + 1000 * m)
    s = {"uniform": 0.0, "zipf0.6": 0.6, "zipf1.0": 1.0, "cluster": 0.8}[dist]
    logit = -s * torch.log(torch.arange(1, E + 1, device="cuda", dtype=torch.float32))
    logit = logit[torch.randperm(E, device="cuda", generator=g)]
    gum = -torch.log(-torch.log(torch.rand(m, E, device="cuda", generator=g).clamp_min(1e-9)))
    if dist == "cluster":  # four consecutive tokens (one request's verify tokens) share expert preferences
        gum = 0.5 * gum + 1.5 * torch.randn((m + 3) // 4, E, device="cuda", generator=g).repeat_interleave(4, 0)[:m]
    ids = (logit + gum).topk(TOPK, dim=1).indices.to(torch.int32).contiguous()
    return ids, torch.softmax(torch.randn(m, TOPK, device="cuda", generator=g), 1).contiguous()


def graph_of(fn, layers):
    fn(0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for l in range(layers):
            fn(l)
    g.replay()
    torch.cuda.synchronize()
    return g


def interleaved_min(graphs, layers, rounds):
    best = {k: float("inf") for k in graphs}
    for _ in range(rounds):
        for k, g in graphs.items():
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            g.replay()
            b.record()
            b.synchronize()
            best[k] = min(best[k], a.elapsed_time(b) * 1e3 / layers)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("m", nargs="*", type=int, default=SERVED_M)
    ap.add_argument("--dists", nargs="+", default=["uniform", "zipf1.0", "cluster"])
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=60)
    a = ap.parse_args()

    import aiter.fused_moe as fm
    from aiter import ActivationType, QuantType
    from aiter.presorted_moe import register_presorted_moe

    mk = lambda n, lo=0, hi=256: torch.randint(lo, hi, (n,), dtype=torch.uint8, device="cuda")
    layers = [MoeWeights.from_shuffled(mk(E * 2 * I * H // 2), mk(E * 2 * I * H // 32, 118, 124), mk(E * H * I // 2),
                                       mk(E * H * 16, 118, 124)) for _ in range(a.layers)]
    fp4 = torch.float4_e2m1fn_x2
    stock_w = []
    for w in layers:
        w13, w2 = w.w13.view(E, 2 * I, H // 2).view(fp4), w.w2.view(E, H, I // 2).view(fp4)
        w13.is_shuffled = w2.is_shuffled = True
        stock_w.append((w13, w2, w.s13.view(E, 2 * I, H // 32).view(torch.float8_e8m0fnu),
                        w.s2.view(E, H, 16).view(torch.float8_e8m0fnu)))
    print("| M | routing | unique | HBM MB | experts+quant us | TB/s | % ceiling | stock fused_moe us | speedup | stock stages (sort given) us | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for m in a.m:
        for dist in a.dists:
            ids, tw = routing(m, dist)
            uniq = int(torch.unique(ids).numel())
            x = (torch.randn(m, H, device="cuda") * 0.5).to(torch.bfloat16)
            r = fm.moe_sorting(ids, tw, E, H, torch.bfloat16, 16, None, None, output_aux=fm.AUX_SORT_OPUS, accumulate=True)
            sort = Sort(r[0], r[1], r[2], r[3], r[6])
            aux = dict(sorted_token_ids=r[0], sorted_weights=r[1], sorted_expert_ids=r[2], num_valid_ids=r[3],
                       m_indices=r[5], reverse_sorted=r[6], topk_ids_ptr=ids.data_ptr(), topk_weights_ptr=tw.data_ptr(),
                       M=m, block_m=16, num_experts=E, topk=TOPK)
            out = torch.empty(m, H, dtype=torch.bfloat16, device="cuda")
            codes, scales = quantize_activations(x)

            def ours(l):
                c, s = quantize_activations(x, codes, scales)
                mxfp4_moe_experts(c, s, layers[l], sort, out)

            def stock_call(l):
                w13, w2, s13, s2 = stock_w[l]
                return fm.fused_moe(x, w13, w2, tw, ids, None, ActivationType.Situv2, QuantType.per_1x32, False, s13, s2,
                                    None, None, beta=4.0, linear_beta=25.0, gate_mode="separated", dtype=torch.bfloat16)

            def stock_presorted(l):
                register_presorted_moe(aux)
                return stock_call(l)

            graphs = {"ours": graph_of(ours, a.layers), "stock": graph_of(stock_call, a.layers),
                      "stock_presorted": graph_of(stock_presorted, a.layers)}
            t = interleaved_min(graphs, a.layers, a.rounds)
            cap = sort_capacity(m)
            nbytes = uniq * EXPERT_BYTES + m * (H // 2 + H // 32) + 2 * m * TOPK * (I // 2 + I // 32) \
                + 2 * m * TOPK * H * 2 + m * H * 2
            tbs = nbytes / t["ours"] / 1e6
            print(f"| {m} | {dist} | {uniq} | {nbytes / 1e6:.0f} | {t['ours']:.1f} | {tbs:.2f} | {tbs / (HBM_CEILING / 1e12) * 100:.0f}% | "
                  f"{t['stock']:.1f} | {t['stock'] / t['ours']:.2f}x | {t['stock_presorted']:.1f} | {t['stock_presorted'] / t['ours']:.2f}x |", flush=True)


if __name__ == "__main__":
    main()
