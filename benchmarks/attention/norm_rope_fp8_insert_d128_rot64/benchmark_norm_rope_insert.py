# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``qkv_norm_rope_insert`` and ``kv_norm_rope_insert_multilayer``
on gfx950.

One HIP graph holds up to 100 launches, each with its own input rows, positions and
slots (``--copies``; fewer when 100 inputs would pass 4 GiB).
The slots of consecutive launches are disjoint, so the cache stores go to new lines.
Inputs smaller than the 256 MB MALL in all stay resident there, as the projection GEMM
that produced them leaves them in serving. Median of 5 rounds of 10 replays, µs per
launch; GB/s counts the input read plus the cache bytes written. The first and the
last launch are checked against ``reference.py`` before timing.

Cases (README): an M3.1 draft layer at TP4 (16 query / 1 KV heads, 5 layers, 128-token
pages); the qkv kernel at the draft's query block (R requests x 4 rows), the multi-layer
kernel at context-row counts from a decode step to a prefill chunk.
"""

import argparse
import statistics

import torch
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.kv_norm_rope_insert_multilayer import (
    kv_norm_rope_insert_multilayer,
)
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.qkv_norm_rope_insert import qkv_norm_rope_insert
from amd_kernels.attention.norm_rope_fp8_insert_d128_rot64.reference import (
    D,
    TOK_B,
    check_bf16,
    check_cache,
    kv_norm_rope_insert_multilayer_reference,
    qkv_norm_rope_insert_reference,
)

EPS = 1e-6
LAYERS = 5
CS_ROWS = 65536
INPUT_BUDGET = 4 << 30
QKV_CASES = [
    ("qkv_tp4_p128", 16, 1, 128, [4, 64, 256, 1024]),
]
ML_CASES = [
    ("multilayer_tp4_p128", 1, 128, [4, 384, 4096]),
]


def timeit_graph(calls, replays=10, rounds=5, warmup=3):
    """Median over rounds of the mean time per call (ms) of one graph of ``calls``."""
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


def copies_for(bytes_per_call, wanted):
    return max(4, min(wanted, INPUT_BUDGET // bytes_per_call))


def rope_cache(gen):
    ang = torch.rand((CS_ROWS, 32), generator=gen, device="cuda", dtype=torch.float32) * 6.3
    return torch.cat([ang.cos(), ang.sin()], dim=1).bfloat16()


def bench_qkv(name, nq, nkv, page, n, wanted, scales):
    ks, vs = scales
    gen = torch.Generator(device="cuda").manual_seed(n * 10 + nq)
    row = (nq + 2 * nkv) * D
    copies = copies_for(n * row * 2, wanted)
    slots_total = max(n * copies, 65536)
    n_pages = -(-slots_total // page)
    cache = torch.zeros((n_pages, nkv, page, TOK_B), dtype=torch.uint8, device="cuda")
    perm = torch.randperm(n_pages * page, generator=gen, device="cuda")
    qw = (torch.randn(D, generator=gen, device="cuda") * 0.5).bfloat16()
    kw = (torch.randn(D, generator=gen, device="cuda") * 0.5).bfloat16()
    cs = rope_cache(gen)
    inputs = [
        (
            (torch.randn((n, row), generator=gen, device="cuda") * 2).bfloat16(),
            torch.randint(0, CS_ROWS, (n,), generator=gen, device="cuda"),
            perm[i * n : (i + 1) * n].contiguous(),
        )
        for i in range(copies)
    ]
    q = torch.empty((n, nq * D), dtype=torch.bfloat16, device="cuda")

    def call(i):
        qkv, pos, slots = inputs[i]
        qkv_norm_rope_insert(qkv, pos, qw, kw, cs, EPS, nq, nkv, q, cache, slots, ks, vs)

    for i in (0, copies - 1):
        before = cache.clone()
        qkv, pos, slots = inputs[i]
        q_ref, (c_ref, mask) = qkv_norm_rope_insert_reference(
            qkv, pos, qw, kw, cs, EPS, nq, nkv, before, slots, ks, vs
        )
        call(i)
        torch.cuda.synchronize()
        check_bf16(q, q_ref, f"{name} n={n}")
        check_cache(cache, c_ref, before, mask, f"{name} n={n}")
    us = timeit_graph([lambda i=i: call(i) for i in range(copies)]) * 1e3
    moved = n * row * 2 + n * nq * D * 2 + n * nkv * TOK_B
    print(f"{name:20s} tokens={n:6d} copies={copies:3d}: {us:7.2f} us, {moved / us / 1e3:6.0f} GB/s")


def bench_multilayer(name, nkv, page, n, wanted):
    gen = torch.Generator(device="cuda").manual_seed(n + nkv)
    row = LAYERS * 2 * nkv * D
    copies = copies_for(n * row * 2, wanted)
    slots_total = max(n * copies, 65536)
    n_pages = -(-slots_total // page)
    caches = [torch.zeros((n_pages, nkv, page, TOK_B), dtype=torch.uint8, device="cuda") for _ in range(LAYERS)]
    perm = torch.randperm(n_pages * page, generator=gen, device="cuda")
    weights = (torch.randn((LAYERS, D), generator=gen, device="cuda") * 0.5).bfloat16()
    cs = rope_cache(gen)
    ks = [torch.tensor(0.75 + 0.1 * i, device="cuda") for i in range(LAYERS)]
    vs = [torch.tensor(1.25 - 0.1 * i, device="cuda") for i in range(LAYERS)]
    inputs = [
        (
            (torch.randn((n, row), generator=gen, device="cuda") * 2).bfloat16(),
            torch.randint(0, CS_ROWS, (n,), generator=gen, device="cuda"),
            perm[i * n : (i + 1) * n].contiguous(),
        )
        for i in range(copies)
    ]

    def call(i):
        kv, pos, slots = inputs[i]
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, [slots] * LAYERS, ks, vs, EPS)

    for i in (0, copies - 1):
        before = [c.clone() for c in caches]
        kv, pos, slots = inputs[i]
        refs = kv_norm_rope_insert_multilayer_reference(
            kv, pos, weights, cs, before, [slots] * LAYERS, ks, vs, EPS
        )
        call(i)
        torch.cuda.synchronize()
        for lay, (got, (ref, mask), b) in enumerate(zip(caches, refs, before)):
            check_cache(got, ref, b, mask, f"{name} n={n} layer {lay}")
        del before, refs
    us = timeit_graph([lambda i=i: call(i) for i in range(copies)]) * 1e3
    moved = n * row * 2 + n * LAYERS * nkv * TOK_B
    print(f"{name:20s} rows={n:6d} copies={copies:3d}: {us:7.2f} us, {moved / us / 1e3:6.0f} GB/s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=100, help="launches (inputs) per graph, at most")
    args = parser.parse_args()
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("requires gfx950")
    scales = torch.tensor(1.0, device="cuda"), torch.tensor(1.3, device="cuda")
    with torch.inference_mode():
        for name, nq, nkv, page, ns in QKV_CASES:
            for n in ns:
                bench_qkv(name, nq, nkv, page, n, args.copies, scales)
        for name, nkv, page, ns in ML_CASES:
            for n in ns:
                bench_multilayer(name, nkv, page, n, args.copies)


if __name__ == "__main__":
    main()
