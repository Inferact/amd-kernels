# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MoE block vLLM runs at TP2 without this entry, timed like ``benchmark_shared_moe.py`` on the same synthetic
weights: ``GateLinear``'s ROCm router GEMM (FP32 kernel to 32 rows, BF16x3 from 2048), ``fused_topk_bias`` (AITER biased top-k), AITER
``fused_moe`` W4A4 (MXFP4, SwiGLU-OAI, separated gate / up, shuffled weights and e8m0), the shared expert on vLLM's
side stream (Triton ``dot_scaled`` MXFP8 gate_up, fused SwiGLU-OAI + MXFP8 quant, ``dot_scaled`` down), the routed
scale and the sum. Needs vLLM and AITER (``VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1``)."""
import argparse
import json
import os
import statistics

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MOE", "1")

import torch
from tests.moe.mxfp4_h6144_i1536_e128_top4.workload import H, I, make_activation, make_experts

SCALE = 2.0


def weights():
    from aiter.utility.fp4_utils import e8m0_shuffle
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.layers.fused_moe.router import bf16x3_router_gemm_rocm

    (w13, s13, w2, s2), _ = make_experts()
    e = w13.shape[0]
    fp4 = torch.float4_e2m1fn_x2
    sw13, sw2 = rocm_aiter_ops.shuffle_weights(w13.view(fp4), w2.view(fp4))
    s13 = e8m0_shuffle(s13.reshape(e * 2 * I, -1)).view(e, 2 * I, -1)
    s2 = e8m0_shuffle(s2.reshape(e * H, -1)).view(e, H, -1)
    g = torch.Generator(device="cuda").manual_seed(607)

    def fp8(n, k):
        q = (torch.randn((n, k), device="cuda", generator=g) * 20).to(torch.float8_e4m3fn)
        return q, torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)

    gate = torch.randn((128, H), device="cuda", generator=g) / H**.5
    bias = torch.randn((128,), device="cuda", generator=g) * .01
    return dict(gate=gate, gate_split=bf16x3_router_gemm_rocm.split_bf16x3(gate), bias=bias, w13=sw13, w2=sw2, s13=s13.view(torch.uint8), s2=s2.view(torch.uint8),
                sh13=fp8(2 * I, H), sh2=fp8(H, I))


def block(x, w, side):
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.model_executor.kernels.linear.mxfp8.rocm_native import (_mxfp8_dot_scaled_gemm,
                                                                         _mxfp8_dot_scaled_linear)
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import fused_topk_bias
    from vllm.model_executor.layers.fused_moe.router.gate_linear import rocm_bf16x3_router_gemm_dispatch_impl
    from vllm.models.minimax_m3.amd.ops.swiglu_oai import swiglu_oai_quantize_mxfp8

    current = torch.cuda.current_stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        gu = _mxfp8_dot_scaled_linear(x, *w["sh13"])
        q, s = swiglu_oai_quantize_mxfp8(gu, alpha=1.702, beta=1.0, limit=7.0, block_m=16)
        shared = _mxfp8_dot_scaled_gemm(q, s, *w["sh2"], torch.bfloat16)
    logits = rocm_bf16x3_router_gemm_dispatch_impl(x, w["gate"], w["gate_split"])
    wt, ids = fused_topk_bias(x, logits, "sigmoid", w["bias"], 4, True)
    routed = rocm_aiter_ops.fused_moe(
        x, w["w13"], w["w2"], wt, ids, quant_method=3,
        activation_method=rocm_aiter_ops.get_aiter_activation_type("swiglu"),
        w1_scale=w["s13"], w2_scale=w["s2"], output_dtype=torch.bfloat16, gate_mode="separated",
        swiglu_limit=7.0)
    current.wait_stream(side)
    return routed * SCALE + shared


def measure(w, m, args, side):
    copies = args.copies or (100 if m <= 2048 else 16)
    xs = [make_activation(m, seed=1000 * m + i) for i in range(copies)]
    block(xs[0], w, side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for x in xs:
            block(x, w, side)
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.rounds):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (args.replays * copies))
    return {"tokens": m, "copies": copies, "median_us": statistics.median(samples),
            "range_us": [min(samples), max(samples)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--copies", type=int)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--replays", type=int, default=10)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    print(json.dumps({"gpu": props.name, "pci": props.pci_bus_id, "torch": torch.__version__,
                      "scope": "vLLM TP2 fallback MoE block: router + AITER W4A4 + Triton MXFP8 shared",
                      "mode": "graph"}))
    w = weights()
    side = torch.cuda.Stream()
    for m in args.tokens:
        print(json.dumps(measure(w, m, args, side)), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
