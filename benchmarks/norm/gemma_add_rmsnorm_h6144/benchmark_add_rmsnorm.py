# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the one-GPU add + RMSNorm kernels on gfx950.

Timing (``timeit_graph``): one HIP graph
cycles over up to 100 input sets (x, residual); the total
footprint is capped at ~2 GB, so at the large token counts fewer sets rotate (each
set is then above the 256 MB MALL anyway). In serving the producer just wrote
these activations, so at small M they sit in the MALL here too. Median of 5
rounds of 10 replays, the time per call. Every case is checked against
``reference.py`` first.
"""

import argparse
import statistics

import torch
from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import H

from tests.norm.gemma_add_rmsnorm_h6144.test_add_rmsnorm import (
    EPS,
    check_mxfp8,
    check_out,
    make_rows,
    make_weight,
)
from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import (
    add_rmsnorm_reference,
    unshuffle_scales,
    untile_codes,
)

TOKENS = [1, 8, 32, 64, 128, 256, 512, 1024, 4096]
MAX_SETS, FOOTPRINT = 100, 2 << 30
TILE = 16  # the decode QKV / gate_up GEMM's code tile at these rows


def timeit_graph(calls, inner=2, replays=10, rounds=5, warmup=5):
    """Median over rounds of the mean time per call (ms) of one graph that cycles
    ``inner`` times over ``calls``."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
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
    return statistics.median(
        s.elapsed_time(e) / (replays * len(calls) * inner) for s, e in events
    )


def make_sets(m):
    per_set = m * H * 4
    n = max(2, min(MAX_SETS, FOOTPRINT // per_set))
    gen = torch.Generator().manual_seed(m)
    sets = []
    for _ in range(n):
        x, r = make_rows(m, gen, "randn"), make_rows(m, gen, "randn")
        sets.append((x, r))
    return sets, make_weight(gen)


def check(m, sets, w):
    x, r = sets[0]
    y, _ = add_rmsnorm_reference(x.cpu(), r.cpu(), w.cpu(), EPS)
    check_out(f"M={m} bf16", an.gemma_add_rmsnorm(x, r, w, EPS)[0], y)
    q, s, _ = an.gemma_add_rmsnorm_mxfp8(x, r, w, EPS, TILE)
    check_mxfp8(f"M={m} mxfp8", untile_codes(q, m), unshuffle_scales(s, m), y)
    q, s, _ = an.gemma_add_rmsnorm_mxfp8_row_major(x, r, w, EPS)
    check_mxfp8(f"M={m} mxfp8 row-major", q.view(torch.uint8), unshuffle_scales(s, m), y)


def flydsl_cases(sets, w):
    return {
        "bf16": [lambda s=s: an.gemma_add_rmsnorm(s[0], s[1], w, EPS) for s in sets],
        "mxfp8_tiled": [lambda s=s: an.gemma_add_rmsnorm_mxfp8(s[0], s[1], w, EPS, TILE) for s in sets],
        "mxfp8_row_major": [lambda s=s: an.gemma_add_rmsnorm_mxfp8_row_major(s[0], s[1], w, EPS) for s in sets],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="*", default=TOKENS)
    args = parser.parse_args()
    if not an.available():
        raise RuntimeError("requires gfx950 with FlyDSL")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU pci_bus {props.pci_bus_id:#x}, free {torch.cuda.mem_get_info()[0] / 2**30:.0f} GiB")
    for m in args.tokens:
        sets, w = make_sets(m)
        check(m, sets, w)
        for name, calls in flydsl_cases(sets, w).items():
            print(f"M={m} sets={len(sets)} {name}: {timeit_graph(calls) * 1e3:.2f} us", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
