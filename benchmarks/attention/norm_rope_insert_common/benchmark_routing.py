# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the norm + RoPE + insert entries add over a direct call of the shape package: CUDA graph replay us per call
through the entry and direct (the graph holds the same kernel), and the host us of the entry's dispatch and of its
reason per eager call (the routed package function replaced by a no-op). Min of ``ROUNDS`` rounds.

    python benchmarks/attention/norm_rope_insert_common/benchmark_routing.py
"""

import importlib
import time

import torch

from amd_kernels.attention import norm_rope_insert_common as E
from tests.attention.norm_rope_insert_common import cases

ROUNDS, REPLAYS, HOST_CALLS = 15, 10, 20000

CASES = [
    ("qkv 32 rows, fp8", lambda: cases.qkv(32, kind="fp8"), "qkv_norm_rope_insert"),
    ("qkv 192 rows, fp8", lambda: cases.qkv(192, kind="fp8"), "qkv_norm_rope_insert"),
    ("multilayer 40 rows, fp8", lambda: cases.multilayer(40, kind="fp8"), "kv_norm_rope_insert_multilayer"),
    ("multilayer 240 rows, fp8", lambda: cases.multilayer(240, kind="fp8"), "kv_norm_rope_insert_multilayer"),
]


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


@torch.inference_mode()
def main():
    print("| call | direct graph us | entry graph us | entry dispatch host us | reason host us |")
    print("|---|---|---|---|---|")
    for name, build, fn_name in CASES:
        direct = replay_us(build()[0], False)
        entry = replay_us(build()[0], True)
        _, args = build()
        mod = importlib.import_module(f"{cases.PKG}.{fn_name}")
        entry_op, reason = getattr(E, fn_name), getattr(E, fn_name + "_blocker")
        real = getattr(mod, fn_name)
        setattr(mod, fn_name, lambda *a, **k: None)
        try:
            dispatch = host_us(lambda op=entry_op, a=args: op(*a)) - host_us(
                lambda m=mod, f=fn_name, a=args: getattr(m, f)(*a)
            )
        finally:
            setattr(mod, fn_name, real)
        why = host_us(lambda r=reason, a=args: r(*a))
        print(f"| {name} | {direct:.2f} | {entry:.2f} | {dispatch:.2f} | {why:.2f} |")


if __name__ == "__main__":
    main()
