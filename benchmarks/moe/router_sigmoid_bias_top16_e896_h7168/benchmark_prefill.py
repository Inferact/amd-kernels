# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Time selection and MXFP4 quantization on identical inputs, in microseconds."""
import argparse
import statistics

import torch
from aiter import biased_grouped_topk, dtypes
from aiter.ops.quant import per_1x32_mx_quant_hip
from amd_kernels.moe import prefill_top16_mxfp4


def separate(logits, latent, bias):
    weights = logits.new_empty((len(logits), 16))
    ids = weights.new_empty(weights.shape, dtype=torch.int32)
    biased_grouped_topk(logits.contiguous(), bias, weights, ids, 1, 1, True, 1.0)
    codes, scales = per_1x32_mx_quant_hip(latent.bfloat16().contiguous(), quant_dtype=dtypes.fp4x2, shuffle=False)
    return weights, ids, codes, scales


def graph(fn, samples, bias):
    for g, l in samples:
        fn(g, l, bias)
    torch.cuda.synchronize()
    result = torch.cuda.CUDAGraph()
    with torch.cuda.graph(result):
        for g, l in samples:
            fn(g, l, bias)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(714)
    bias = torch.randn(896, device="cuda") * .12
    for m in args.rows:
        samples = []
        for _ in range(8):
            if args.packed:
                samples.append(torch.randn(m, 4480, device="cuda").split((896, 3584), 1))
            else:
                samples.append((torch.randn(m, 896, device="cuda"), torch.randn(m, 3584, dtype=torch.bfloat16, device="cuda")))
        graphs = [graph(f, samples, bias) for f in (separate, prefill_top16_mxfp4)]
        times = [[], []]
        for r in range(9):
            for i in (r % 2, 1 - r % 2):
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(10):
                    graphs[i].replay()
                end.record()
                end.synchronize()
                times[i].append(start.elapsed_time(end) * 1000 / (10 * len(samples)))
        a, b = map(statistics.median, times)
        print(f"rows={m} packed={args.packed} separate_us={a:.2f} fused_us={b:.2f} speedup={a/b:.3f}", flush=True)


if __name__ == "__main__":
    main()
