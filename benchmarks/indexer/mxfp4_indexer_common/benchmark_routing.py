# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the indexer entries add to a call: graph replay of 20 calls through the entry against 20 direct calls of the
routed package (min of 15 rounds x 5 replays, each graph timed twice interleaved, us per call), and the host time of
the entry's dispatch (the routed function replaced by a no-op) and of its reason, us per call (min of 15 rounds of
2000).
Run: python3 benchmarks/indexer/mxfp4_indexer_common/benchmark_routing.py
"""

import importlib
import sys
import time
from pathlib import Path

import torch

from amd_kernels.indexer.mxfp4_indexer_common import paged, prefill

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.indexer.mxfp4_paged_score_topk_d128_h32_top2048.inputs import make_inputs as paged_inputs
from tests.indexer.mxfp4_prefill_score_d128_h32.inputs import make_inputs as score_inputs
from tests.indexer.mxfp4_prefill_topk_d128_h32_top2048.inputs import make_logits, make_windows

CALLS = 20


def replay_us(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / (5 * CALLS))
    return best


def host_us(fn, n=2000):
    best = float("inf")
    for _ in range(15):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t) * 1e6 / n)
    return best


def dispatch_us(module, name, call):
    """Host time of ``call`` with the routed package's ``name`` replaced by a no-op."""
    real = getattr(module, name)
    setattr(module, name, lambda *a, **k: None)
    try:
        return host_us(call)
    finally:
        setattr(module, name, real)


def cases():
    pkg = importlib.import_module(paged._ROUTES.hosts[(32, 128, 2048, 64)])
    for seqs, qlen, width in ((2, 6, 20480), (13, 5, 32768), (48, 4, 16384)):
        inputs = paged_inputs(seqs, qlen, width)
        rows = seqs * qlen
        out = dict(out_indices=torch.empty(rows, 2048, dtype=torch.int32, device="cuda"),
                   out_scores=torch.empty(rows, 2048, dtype=torch.float32, device="cuda"))
        yield (f"mxfp4_paged_score_topk {seqs} x {qlen}, {width} keys", pkg, "mxfp4_paged_score_topk",
               lambda i=inputs, o=out: paged.mxfp4_paged_score_topk(**i, **o),
               lambda i=inputs, o=out: pkg.mxfp4_paged_score_topk(**i, **o),
               lambda i=inputs, o=out: paged.mxfp4_paged_score_topk_blocker(**i, **o))
    pkg = importlib.import_module(prefill._SCORE_ROUTES.hosts[(32, 128)])
    for rows, keys in ((512, 32768), (2048, 70000)):
        args = score_inputs(rows, keys)
        args = (*args, torch.empty(rows, args[3].shape[0] * 64, dtype=torch.float32, device="cuda"), 1.0)
        yield (f"mxfp4_prefill_score {rows} rows, {keys} keys", pkg, "score_rows",
               lambda a=args: prefill.mxfp4_prefill_score(*a), lambda a=args: pkg.score_rows(*a),
               lambda a=args: prefill.mxfp4_prefill_score_blocker(*a))
    pkg = importlib.import_module(prefill._TOPK_ROUTES.hosts[(2048,)])
    for rows, width in ((256, 60000), (1024, 131072)):
        ks, ke = make_windows(rows, width, "causal")
        args = (make_logits("gauss", rows, width), ks, ke, torch.empty(rows, 2048, dtype=torch.int32, device="cuda"))
        yield (f"mxfp4_prefill_topk {rows} rows, {width} keys", pkg, "topk_rows",
               lambda a=args: prefill.mxfp4_prefill_topk(*a), lambda a=args: pkg.topk_rows(*a),
               lambda a=args: prefill.mxfp4_prefill_topk_blocker(*a))


def main():
    print("| call | direct us | entry us | added us | entry dispatch host us | reason host us |")
    print("|---|---|---|---|---|---|")
    for label, module, name, entry, direct, reason in cases():
        d, e = replay_us(direct), replay_us(entry)
        d, e = min(d, replay_us(direct)), min(e, replay_us(entry))  # interleaved: no order bias
        print(f"| {label} | {d:.2f} | {e:.2f} | {e - d:+.2f} | {dispatch_us(module, name, entry):.2f} | "
              f"{host_us(reason):.2f} |", flush=True)


if __name__ == "__main__":
    main()
