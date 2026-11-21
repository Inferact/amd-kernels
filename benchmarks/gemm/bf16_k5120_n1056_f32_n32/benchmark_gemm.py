# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``gemm`` against the two launches it replaces (the baseline): ``torch.mm(x, wa^T,
out_dtype=float32)`` and ``F.linear(x, wb)`` (bf16), on gfx950. One HIP graph holds up to 100 launches, each with its own
activation rows; the weights are shared (and so stay in the 256 MB MALL between launches, as a layer's weights do not:
``--weights N`` cycles through N copies of the weights). Rounds run A B B A (kernel, baseline, baseline, kernel); each is
the median of 5 x 10 replays, us per launch.
"""

import argparse
import statistics

import torch
import torch.nn.functional as F
from amd_kernels.gemm.bf16_k5120_n1056_f32_n32 import gemm, shuffle_weight

ROWS = [16, 48, 96, 192, 384, 768]
SPLITS = [(1024, 32), (512, 32), (0, 32)]


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


def bench(m, na, nb, wanted, n_weights):
    gen = torch.Generator(device="cuda").manual_seed(m + na)
    copies = max(4, min(wanted, (4 << 30) // (m * 5120 * 2)))
    xs = [torch.randn((m, 5120), generator=gen, device="cuda").bfloat16() for _ in range(copies)]
    ws = [(torch.randn((na + nb, 5120), generator=gen, device="cuda") * 0.03).bfloat16() for _ in range(n_weights)]
    shuf = [shuffle_weight(w[:na] if na else None, w[na:]) for w in ws]
    out_a = torch.empty((m, na), dtype=torch.float32, device="cuda") if na else None
    out_b = torch.empty((m, nb), dtype=torch.bfloat16, device="cuda")

    def run_a(i):
        gemm(xs[i], shuf[i % n_weights], out_a, out_b)

    def run_b(i):
        w = ws[i % n_weights]
        if na:
            torch.mm(xs[i], w[:na].T, out_dtype=torch.float32)
        F.linear(xs[i], w[na:])

    run_a(0)
    ref = F.linear(xs[0], ws[0][na:])
    torch.cuda.synchronize()
    assert (out_b.float() - ref.float()).abs().max() <= 2e-2 * ref.float().abs().max()
    times = {"kernel": [], "baseline": []}
    for name in ("kernel", "baseline", "baseline", "kernel"):
        fn = run_a if name == "kernel" else run_b
        times[name].append(timeit_graph([lambda i=i, fn=fn: fn(i) for i in range(copies)]) * 1e3)
    k, b = statistics.mean(times["kernel"]), statistics.mean(times["baseline"])
    print(
        f"n={na}+{nb} rows={m:4d} copies={copies:3d}: kernel {k:7.2f} us (runs {times['kernel'][0]:.2f} / "
        f"{times['kernel'][1]:.2f}), baseline {b:7.2f} us (runs {times['baseline'][0]:.2f} / {times['baseline'][1]:.2f}), "
        f"{b / k:4.2f}x"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--weights", type=int, default=8)
    parser.add_argument("--rows", type=int, nargs="*", default=ROWS)
    args = parser.parse_args()
    with torch.inference_mode():
        for na, nb in SPLITS:
            for m in args.rows:
                bench(m, na, nb, args.copies, args.weights)
