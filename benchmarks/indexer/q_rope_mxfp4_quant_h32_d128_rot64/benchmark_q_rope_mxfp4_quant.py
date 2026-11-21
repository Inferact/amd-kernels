# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``q_rope_mxfp4_quant`` against AITER's ``q_rope_mxfp4_quant`` (the baseline) on gfx950.

One HIP graph holds up to 100 launches, each with its own query, positions and weights (fewer when the inputs would
pass 4 GiB). Rounds run A B B A (kernel, baseline, baseline, kernel); each is the median of 5 x 10 replays, us per
launch. The first and last launch of the kernel are checked against the baseline's bytes before timing.
"""

import argparse
import statistics

import torch
from aiter.ops.triton.rope.q_rope_mxfp4_quant import q_rope_mxfp4_quant as aiter_op
from amd_kernels.indexer.q_rope_mxfp4_quant_h32_d128_rot64 import q_rope_mxfp4_quant

ROWS = [16, 48, 96, 192, 384, 768]
WSCALE = 0.0883883476483184 * 32**-0.5
INPUT_BUDGET = 4 << 30


def timeit_graph(calls, replays=10, rounds=5, warmup=3):
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        graph.replay()
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / (replays * len(calls)) for s, e in events)


def bench(n, wanted):
    gen = torch.Generator(device="cuda").manual_seed(n)
    copies = max(4, min(wanted, INPUT_BUDGET // (n * 32 * 128 * 2)))
    cs = torch.randn((65536, 64), generator=gen, device="cuda")
    inputs = [
        (
            (torch.randn((n, 32, 128), generator=gen, device="cuda") * 2).bfloat16(),
            torch.randint(0, 65536, (n,), generator=gen, device="cuda"),
            torch.randn((n, 32), generator=gen, device="cuda").bfloat16(),
        )
        for _ in range(copies)
    ]
    outs = (
        torch.empty((n, 32, 64), dtype=torch.uint8, device="cuda"),
        torch.empty((n, 32, 4), dtype=torch.uint8, device="cuda"),
        torch.empty((n, 32), dtype=torch.float32, device="cuda"),
    )

    def run_a(i):
        q, pos, w = inputs[i]
        q_rope_mxfp4_quant(q, pos, cs, w, WSCALE, *outs)

    def run_b(i):
        q, pos, w = inputs[i]
        aiter_op(q, pos, cs, w, WSCALE)

    for i in (0, copies - 1):
        run_a(i)
        torch.cuda.synchronize()
        q, pos, w = inputs[i]
        for g, r in zip(outs, aiter_op(q, pos, cs, w, WSCALE)):
            assert torch.equal(g, r)
    times = {"kernel": [], "baseline": []}
    for name in ("kernel", "baseline", "baseline", "kernel"):
        fn = run_a if name == "kernel" else run_b
        times[name].append(timeit_graph([lambda i=i, fn=fn: fn(i) for i in range(copies)]) * 1e3)
    k, b = statistics.mean(times["kernel"]), statistics.mean(times["baseline"])
    print(
        f"rows={n:4d} copies={copies:3d}: kernel {k:7.2f} us (runs {times['kernel'][0]:.2f} / {times['kernel'][1]:.2f}), "
        f"baseline {b:7.2f} us (runs {times['baseline'][0]:.2f} / {times['baseline'][1]:.2f}), {b / k:4.2f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--rows", type=int, nargs="*", default=ROWS)
    args = parser.parse_args()
    with torch.inference_mode():
        for n in args.rows:
            bench(n, args.copies)
