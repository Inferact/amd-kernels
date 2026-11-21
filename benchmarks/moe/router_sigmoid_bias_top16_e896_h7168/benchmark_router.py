# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the MoE router (GEMM + sigmoid + bias + top-16, one launch) against vLLM's path: GateLinear's
fp32-out bf16 GEMM (hipBLASLt) followed by AITER's biased_grouped_topk (one group, topk_reg_kernel). One graph of 24 calls
over 24 weight copies (24 x 12.8 MB > the 256 MB MALL: HBM on every call, as in serving) and 24 activation sets, 5 replays
x 15 rounds, minimum us (min of 15). Floors measured on the same kernel (the ``skel`` bits of ``compile_router_e896_h7168``):
``stream`` = the weight stream alone (grid, DMAs and launch of the real kernel; no activations, MFMAs, stores, consumers),
``prod`` = every workgroup's GEMM and slab stores, no consumers, ``skel`` = the whole hand-off (slab stores, the token
gather, the exchange) with no MFMAs and no top-16. The % columns are fused against ``skel`` (the measured ceiling of
this design's critical path before the top-16 chain) and against the HBM roofline (7.0 TB/s measured
nontemporal reads + one launch). Peak: 1.98 PF measured bf16 MFMA.
Run: python3 benchmarks/moe/router_sigmoid_bias_top16_e896_h7168/benchmark_router.py [M ...]
"""
import os
import statistics
import sys

import torch

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168 import host, router_sigmoid_bias_top16_e896_h7168
from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168.reference import E, H, TOPK

HBM_TBPS = 7.0
PEAK_BF16_TFLOPS = 1980.0
COPIES = 24
LAST_MEDIAN = 0.0  # the median of the rounds of the last ``timing`` (which returns the minimum)


def timing(calls):
    for f in calls[:2]:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / (5 * len(calls)))
    global LAST_MEDIAN
    LAST_MEDIAN = statistics.median(times)
    return min(times)


if __name__ == "__main__":
    from aiter import biased_grouped_topk
    import flydsl.compiler as flyc
    from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168.kernel import compile_router_e896_h7168

    def skeleton(m, xs, bits):
        mt = host.m_tile_fragments(m)
        cfg = host._config(mt)
        part, flags = host._workspace(xs[0].device, torch.cuda.current_stream().cuda_stream, mt, cfg)
        ow, oi = torch.empty(m, TOPK, device="cuda"), torch.empty(m, TOPK, dtype=torch.int32, device="cuda")
        exe = compile_router_e896_h7168(mt, cfg[0], 1, 1, cfg[1], cfg[2], 0, 0, bits, cfg[3])
        mk = lambda x, w: (x.data_ptr(), w.data_ptr(), part.data_ptr(), flags.data_ptr(), bias.data_ptr(), ow.data_ptr(),
                           oi.data_ptr(), 0, 0, 0, 0, 0, 0, 0, 0, m, TOPK, 0x3f800000, 0, 0)
        exe._cf = flyc.compile(exe, *mk(xs[0], wss[0]), torch.cuda.current_stream())
        return timing([lambda x=x, w=w: exe._cf(*mk(x, w), torch.cuda.current_stream()) for x, w in zip(xs, wss)])

    ms = [int(v) for v in sys.argv[1:]] or [1, 12, 24, 40, 65, 80, 128, 192]
    torch.manual_seed(0)
    ws = [(torch.randn(E, H, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(COPIES)]
    wss = [host.shuffle_weight(w) for w in ws]
    bias = torch.randn(E, device="cuda") * 0.1
    scratch = [torch.zeros(1, device="cuda") for _ in range(COPIES)]
    floor = timing([lambda s=s: s.add_(1) for s in scratch])
    wbytes = E * H * 2
    print(f"launch floor (one trivial kernel in the graph): {floor:.2f} us; us, min of 15")
    print(f"{'M':>4} {'cfg':>8} {'stock':>6} {'fused':>6} {'+aux':>6} {'x':>5} | {'stream':>6} {'prod':>6} {'skel':>6} {'%skel':>5} | "
          f"{'HBM':>5} {'%HBM':>5} {'TFLOPs':>6} {'%peak':>5}")
    for m in ms:
        xs = [torch.randn(m, H, device="cuda").to(torch.bfloat16) for _ in range(COPIES)]
        sw = [torch.empty(m, TOPK, device="cuda") for _ in range(COPIES)]
        si = [torch.empty(m, TOPK, dtype=torch.int32, device="cuda") for _ in range(COPIES)]
        stock = timing([lambda x=x, w=w, a=a, b=b: biased_grouped_topk(torch.mm(x, w.T, out_dtype=torch.float32), bias,
                                                                         a, b, 1, 1, True, 1.0)
                        for x, w, a, b in zip(xs, ws, sw, si)])
        fused = timing([lambda x=x, w=w: router_sigmoid_bias_top16_e896_h7168(x, w, bias, shuffled=True)
                        for x, w in zip(xs, wss)])
        auxes = [host.make_aux(m, "cuda") for _ in range(COPIES)]
        plus_aux = timing([lambda x=x, w=w, a=a: router_sigmoid_bias_top16_e896_h7168(x, w, bias, shuffled=True, aux=a)
                           for x, w, a in zip(xs, wss, auxes)])
        stream = skeleton(m, xs, 4 | 1 | 8 | 16)
        prod = skeleton(m, xs, 8)
        skel = skeleton(m, xs, 1 | 2)
        hbm_us = (wbytes + m * H * 2 + m * TOPK * 8) / (HBM_TBPS * 1e6) + floor
        tflops = 2 * m * E * H / fused / 1e6
        cfg = ":".join(str(v) for v in host._config(host.m_tile_fragments(m)))
        print(f"{m:4d} {cfg:>8} {stock:6.1f} {fused:6.1f} {plus_aux:6.1f} {stock / fused:4.2f}x | {stream:6.1f} {prod:6.1f} "
              f"{skel:6.1f} {100 * skel / fused:5.0f} | {hbm_us:5.2f} {100 * hbm_us / fused:5.0f} {tflops:6.0f} "
              f"{100 * tflops / PEAK_BF16_TFLOPS:5.1f}", flush=True)
