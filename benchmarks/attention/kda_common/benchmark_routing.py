# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the KDA entries add over a direct call of the shape package: CUDA graph replay us per call through the
entry and direct (the graph holds the same kernels), and the host us of the entry's dispatch and of its reason per
eager call (the routed package function replaced by a no-op). Min of ``ROUNDS`` rounds.

    python benchmarks/attention/kda_common/benchmark_routing.py
"""

import importlib
import time

import torch

from amd_kernels.attention import kda_common as E

from tests.attention.kda_common import cases

ROUNDS, REPLAYS, HOST_CALLS = 15, 10, 20000

CASES = [
    ("decode 12 rows (2 x 6)", lambda: cases.decode(12, 6, False), cases.DECODE_PKG, "kda_decode_fb_gate_spec"),
    ("decode 192 rows (32 x 6)", lambda: cases.decode(192, 6, True), cases.DECODE_PKG, "kda_decode_fb_gate_spec"),
    ("prefill 2048 tokens (one launch)", lambda: cases.prefill([2048]), cases.PREFILL_PKG, "prefill"),
    ("prefill 4096 tokens (staged)", lambda: cases.prefill([4096]), cases.PREFILL_PKG, "prefill"),
    ("mono 12 rows (2 x 6)", lambda: cases.mono(12, 6), cases.MONO_PKG, "kda_mono_decode_k1"),
    ("mono 16 rows (4 x 4)", lambda: cases.mono(16, 4), cases.MONO_PKG, "kda_mono_decode_k1"),
]  # fmt: skip


def replay_us(call, entry):
    call.run(entry)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(REPLAYS):
            call.run(entry)
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(ROUNDS):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) * 1e3 / REPLAYS)
    return best


def host_us(fn):
    best = float("inf")
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(HOST_CALLS):
            fn()
        best = min(best, (time.perf_counter() - t0) * 1e6 / HOST_CALLS)
    return best


def calls(kind, inputs, pkg):
    """(entry op, direct op, entry reason) of one call, without the step work around it (table, ``begin_step``)."""
    if kind == "decode":
        return (lambda: E.kda_decode_fb_gate_spec(**inputs), lambda: pkg.kda_decode_fb_gate_spec(**inputs),
                lambda: E.kda_decode_fb_gate_spec_blocker(**inputs))
    if kind == "prefill":
        args, kw = inputs
        kw = dict(kw, table=pkg.chunk_table(args[11], args[12], args[9], args[10]))
        return (lambda: E.kda_prefill(*args, **kw), lambda: pkg.prefill(*args, **kw),
                lambda: E.kda_prefill_blocker(*args, **kw))
    ws, args, kw = inputs
    ws = ws[True]
    return (lambda: E.kda_mono_decode(ws, 0, *args, **kw), lambda: pkg.kda_mono_decode_k1(ws, 0, *args, **kw),
            lambda: E.kda_mono_decode_blocker(ws, 0, *args, **kw))


@torch.inference_mode()
def main():
    print("| call | direct graph us | entry graph us | entry dispatch host us | reason host us |")
    print("|---|---|---|---|---|")
    for name, build, pkg_name, fn_name in CASES:
        direct = replay_us(build()[0], False)
        entry = replay_us(build()[0], True)
        _, inputs = build()
        pkg = importlib.import_module(pkg_name)
        entry_op, direct_op, reason = calls(name.split()[0], inputs, pkg)
        real = getattr(pkg, fn_name)
        setattr(pkg, fn_name, lambda *a, **k: None)
        try:
            dispatch = host_us(entry_op) - host_us(direct_op)
        finally:
            setattr(pkg, fn_name, real)
        reason = host_us(reason)
        print(f"| {name} | {direct:.2f} | {entry:.2f} | {dispatch:.2f} | {reason:.2f} |")


if __name__ == "__main__":
    main()
