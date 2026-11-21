# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""mHC seam finisher graph latency per row count and record count, A B B A, against the seam kernel without post-mix
(``norm/mhc_post_pre_rmsnorm_h5120``: reads the residual and does the gate projection itself, the structure the finisher
replaces); both emit the MXFP8 output; 100 distinct inputs per graph, 5 rounds of 5 replays, median us.

  python3 benchmarks/norm/mhc_gate_finish_rmsnorm_h5120/benchmark_finish.py [--ms 16,192]
"""

import argparse
import statistics

import torch

from amd_kernels.norm.mhc_gate_finish_rmsnorm_h5120 import mhc_gate_finish
from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120 import mhc_post_pre_rmsnorm, pack_gate_weight

H, COUNT = 5120, 100
ROWS = (16, 32, 64, 128, 160, 192, 256, 320, 384, 512, 640, 768)
RECORDS = (64, 160)  # 40 column blocks x 1 wave ... 40 x 4 waves: the producer's record counts at the tuned rows
KW = dict(rms_eps=1e-6, hc_pre_eps=1e-6, hc_sinkhorn_eps=1e-6, hc_post_mult=1.5, norm_eps=1e-6)


def capture(fn):
    fn(0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for j in range(COUNT):
            fn(j)
    graph.replay()
    torch.cuda.synchronize()
    return graph


def replay_us(graph, reps=5):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        graph.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * COUNT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ms", default=",".join(map(str, ROWS)))
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    fn_seam = pack_gate_weight(torch.randn(24, 4 * H, generator=gen, device=dev) * 0.02)
    sc, base = torch.tensor([0.5, 0.7, 0.9], device=dev), torch.randn(24, generator=gen, device=dev) * 0.3
    nw = torch.randn(H, generator=gen, device=dev).bfloat16()
    print("rows  seam(no post)  finisher nb=" + "  finisher nb=".join(map(str, RECORDS)))
    for m in map(int, args.ms.split(",")):
        res = [torch.randn(m, 4, H, generator=gen, device=dev).bfloat16() for _ in range(COUNT)]
        x1 = [torch.randn(m, H, generator=gen, device=dev).bfloat16() for _ in range(COUNT)]
        pre = torch.rand(m, 4, generator=gen, device=dev)
        gps = {nb: [torch.rand(m, nb, 32, generator=gen, device=dev) for _ in range(COUNT)] for nb in RECORDS}
        graphs = {"seam": capture(lambda j: mhc_post_pre_rmsnorm(res[j], fn_seam, sc, base, nw, pre_mix=pre, quant=True,
                                                                  **KW))}
        for nb in RECORDS:
            graphs[nb] = capture(lambda j, nb=nb: mhc_gate_finish(gps[nb][j], x1[j], sc, base, nw, quant=True, **KW))
        times = {k: [] for k in graphs}
        order = list(graphs)
        for k in (order + order[::-1]) * args.rounds:
            times[k].append(replay_us(graphs[k]))
        print(f"{m:4d}  " + "  ".join(f"{statistics.median(times[k]):8.2f}" for k in graphs), flush=True)


if __name__ == "__main__":
    main()
