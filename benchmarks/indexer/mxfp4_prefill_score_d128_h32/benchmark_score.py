# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of ``score_rows`` vs AITER's scorer over key sets larger than the LLC; roofline table.

    python3 benchmarks/indexer/mxfp4_prefill_score_d128_h32/benchmark_score.py [json]
"""

import json
import statistics
import sys

import torch
from amd_kernels.indexer.mxfp4_prefill_score_d128_h32 import host
from tests.indexer.mxfp4_prefill_score_d128_h32.inputs import make_inputs

PEAK_FP4_TFLOPS = 8200.0  # measured register-resident 16x16x128 FP4 MFMA
MODEL_TFLOPS = 2440.0  # kernel.py cycle model at 8 rows per workgroup: 464 cycles per row-page at 2.1 GHz on 1024 SIMDs
CASES = [(512, 150_000), (512, 500_000), (715, 150_000), (715, 500_000), (1280, 150_000), (2048, 150_000)]


def replay_us(fn, iters=10, rounds=5):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    return statistics.median(times)


def aiter_fn(inputs, out):
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
        flydsl_pa_mqa_logits_fp4_prefill,
    )

    q_payload, q_scale, weights, key_payload, key_scale, ks, ke = inputs
    pages = key_payload.shape[0]
    rows = q_payload.shape[0]
    table = torch.arange(pages + 8, dtype=torch.int32, device="cuda").unsqueeze(0)
    batches = torch.zeros(rows, dtype=torch.int32, device="cuda")
    _, schedule, n_ctas = compute_prefill_schedule(batches, ks, ke, 256, max(4096, rows), pages * 64)

    def run():
        flydsl_pa_mqa_logits_fp4_prefill(
            q_payload, q_scale, key_payload, key_scale, table, weights, batches, ks, ke, pages * 64,
            cta_info=schedule, n_ctas=n_ctas, out=out,
        )  # fmt: skip

    return run


LLC_BYTES = 256 * 2**20


def main():
    results = []
    print("rows  keys     aiter_us  ours_us  speedup  TFLOPs  %mfma_peak  %model  key_GB_read  logits_GB")
    for rows, ctx in CASES:
        inputs = make_inputs(rows, ctx + rows)
        pages = inputs[3].shape[0]
        sets = [inputs] + [
            inputs[:3] + (torch.randint_like(inputs[3], 256), torch.randint_like(inputs[4], 118, 125)) + inputs[5:]
            for _ in range(max(0, 2 * LLC_BYTES // (pages * 4352) - 1))
        ]
        out = torch.empty(rows, pages * 64, dtype=torch.float32, device="cuda")
        flops = 2.0 * rows * (ctx + rows / 2) * 32 * 128
        aiters = [aiter_fn(s, out) for s in sets]
        t_aiter = replay_us(lambda: [f() for f in aiters]) / len(sets)
        t_ours = replay_us(lambda: [host.score_rows(*s, out) for s in sets]) / len(sets)
        tflops = flops / t_ours / 1e6
        groups, _ = host.grid_for(rows, pages)
        key_gb = groups * (ctx + rows) * 68 / 1e9
        logits_gb = rows * (ctx + rows / 2) * 4 / 1e9
        print(f"{rows:5d} {ctx:7d} {t_aiter:9.0f} {t_ours:8.0f} {t_aiter / t_ours:7.2f}x {tflops:7.0f} {100 * tflops / PEAK_FP4_TFLOPS:8.0f}% {100 * tflops / MODEL_TFLOPS:6.0f}% {key_gb:11.2f} {logits_gb:9.2f}")
        results.append(dict(rows=rows, ctx=ctx, aiter_us=t_aiter, us=t_ours, tflops=tflops))
    if len(sys.argv) > 1:
        json.dump(results, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
