# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""TP2 o_proj N6144 K4096: the public entries versus vLLM's Triton ``tl.dot_scaled`` linear (the TP2 fallback).

Layer 39 o_proj, TP2 rank 0 (weight[:, :4096], scale[:, :128]), 16 rotating weight copies (> 400 MB), seeded
inputs, 15 alternated rounds of 10 graph replays, median us per call. ``--form codes``: the producer's MXFP8
codes (ours: ``mxfp8_gemm_prequant`` in ``prequant_code_tile_rows``' layout; Triton: ``_mxfp8_dot_scaled_gemm`` on
row-major codes). ``--form bf16``: a bf16 activation (``mxfp8_linear`` / ``_mxfp8_dot_scaled_linear``, both
quantize). Roofline: max(bytes / 6.5 TB/s, flops / 5 PF).

  python3 benchmarks/gemm/mxfp8_n6144_k4096/benchmark_n6144_k4096.py --checkpoint /path/to/model --form codes
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref

N, K = 6144, 4096
COPIES = 16
DECODE_MS = "1,8,16,32,48,64,80,96,128,160,192,256,320,384,448,512"
PREFILL_MS = "640,768,1024,1280,1536,2048,2560,3072,4096,6144,8192,16384,32768"


def roofline_us(m):
    nbytes = N * K * 33 / 32 + m * K * 33 / 32 + 2 * m * N
    return max(nbytes / 6.5e6, 2 * m * N * K / 5e9)


def load_weights(checkpoint, layer):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"language_model.model.layers.{layer}.self_attn.o_proj."
    raw = []
    for field, cols in (("weight", K), ("weight_scale", K // 32)):
        with safe_open(str(checkpoint / index[prefix + field]), framework="pt") as f:
            raw.append(f.get_tensor(prefix + field)[:, :cols].contiguous().cuda())
    codes, scales = raw[0].view(torch.float8_e4m3fn), raw[1].view(torch.uint8)
    w, ws = g.preshuffle_mxfp8_weight(codes, scales)
    ours = [(w, ws)] + [(w.clone(), ws.clone()) for _ in range(COPIES - 1)]
    plain = [(codes, scales)] + [(codes.clone(), scales.clone()) for _ in range(COPIES - 1)]
    return ours, plain


def capture(call, count):
    for j in range(count):
        call(j)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(count):
            call(j)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def measure(graph, count):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        graph.replay()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) * 1000 / (10 * count)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=39)
    ap.add_argument("--form", choices=("codes", "bf16"), default="codes")
    ap.add_argument("--ms", default=None, help=f"default: codes {DECODE_MS}; bf16 {DECODE_MS},{PREFILL_MS}")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_native as rn
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import mxfp8_e4m3_quantize

    torch.set_num_threads(1)
    ours_w, plain_w = load_weights(args.checkpoint, args.layer)
    ms = args.ms or (DECODE_MS if args.form == "codes" else DECODE_MS + "," + PREFILL_MS)
    results = {}
    for m in map(int, ms.split(",")):
        count = 32 if m <= 4096 else 16
        xs = [torch.from_numpy(np.random.default_rng(1000 * m + i).standard_normal((m, K), dtype=np.float32))
              .cuda().bfloat16() for i in range(count)]
        if args.form == "codes":
            tile = g.prequant_code_tile_rows(m, N, K)
            prod = [g.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else g.mxfp8_e4m3_quantize_aiter(x) for x in xs]
            plain = [mxfp8_e4m3_quantize(x) for x in xs]
            arms = {
                "ours": lambda j: g.mxfp8_gemm_prequant(*prod[j], *ours_w[j % COPIES], torch.bfloat16, m),
                "triton": lambda j: rn._mxfp8_dot_scaled_gemm(*plain[j], *plain_w[j % COPIES], torch.bfloat16),
            }
        else:
            arms = {
                "ours": lambda j: g.mxfp8_linear(xs[j], *ours_w[j % COPIES]),
                "triton": lambda j: rn._mxfp8_dot_scaled_linear(xs[j], *plain_w[j % COPIES]),
            }
        expected = ref.mxfp8_linear_reference(xs[0], *plain_w[0])
        for name, call in arms.items():
            metrics = ref.compare(call(0), expected)
            assert metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX, (m, name, metrics)
        graphs = {name: capture(call, count) for name, call in arms.items()}
        rounds = {name: [] for name in arms}
        for r in range(15):
            for name in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
                rounds[name].append(measure(graphs[name], count))
        med = {name: statistics.median(v) for name, v in rounds.items()}
        ideal = roofline_us(m)
        config = g._select_prequant_config(m, N, K, torch.bfloat16) if args.form == "codes" else \
            g._select_config(m, N, K, torch.bfloat16)
        results[m] = dict(us=med["ours"], triton_us=med["triton"], ideal_us=ideal,
                          fraction=ideal / med["ours"], config=repr(config), rounds=rounds)
        print(f"M {m:6d}  ours {med['ours']:8.2f} us  triton {med['triton']:8.2f} us  "
              f"x{med['triton'] / med['ours']:.2f}  roofline {ideal:7.2f} us ({100 * ideal / med['ours']:4.1f}%)  "
              f"{config.kernel} {config.tile_m}x{config.tile_n}", flush=True)
        if args.out:
            args.out.write_text(json.dumps(results, indent=2) + "\n")
        del graphs, xs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
