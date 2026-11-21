# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``rope_fp8_kv_insert`` against vLLM's fused C op (the baseline) on gfx950.

One HIP graph holds up to 100 launches, each with its own q / KV rows, positions and slots (fewer when the inputs
would pass 4 GiB); the slots of consecutive launches are disjoint. Rounds run A B B A (kernel, baseline, baseline,
kernel); each is the median of 5 x 10 replays, us per launch. The first and last launch of the kernel are checked
against the baseline's bytes before timing.
"""

import argparse
import statistics

import torch
from amd_kernels.attention.rope_fp8_kv_insert_h64_d512_rot64 import rope_fp8_kv_insert

ROWS = [16, 48, 96, 192, 384, 768]
BS = 128
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
    import vllm._custom_ops  # noqa: F401

    base = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert
    gen = torch.Generator(device="cuda").manual_seed(n)
    copies = max(4, min(wanted, INPUT_BUDGET // (n * 64 * 512 * 3)))
    pages = -(-max(n * copies, 65536) // BS)
    cs = torch.randn((65536, 64), generator=gen, device="cuda")
    perm = torch.randperm(pages * BS, generator=gen, device="cuda")
    ks = torch.ones(1, device="cuda")
    qsi = torch.ones(1, device="cuda")
    cache_a = torch.zeros((pages, BS, 512), dtype=torch.float8_e4m3fn, device="cuda")
    cache_b = torch.zeros_like(cache_a)
    qa = torch.empty((n, 64, 512), dtype=torch.float8_e4m3fn, device="cuda")
    qb = torch.empty_like(qa)
    inputs = [
        (
            (torch.randn((n, 64, 512), generator=gen, device="cuda") * 2).bfloat16(),
            (torch.randn((n, 512), generator=gen, device="cuda") * 2).bfloat16(),
            torch.randint(0, 65536, (n,), generator=gen, device="cuda"),
            perm[i * n : (i + 1) * n].contiguous(),
        )
        for i in range(copies)
    ]

    def run_a(i):
        q, kv, pos, slots = inputs[i]
        rope_fp8_kv_insert(q, kv, qa, cache_a, slots, pos, cs, ks, qsi)

    def run_b(i):
        q, kv, pos, slots = inputs[i]
        base(q, kv, qb, cache_b, slots, pos, cs, ks, qsi, 1e-6, BS, False)

    for i in (0, copies - 1):
        run_a(i)
        run_b(i)
        torch.cuda.synchronize()
        d = qa.view(torch.uint8) != qb.view(torch.uint8)
        assert not d.any(), (i, d.sum().item(), d.nonzero()[:4].tolist())
    assert torch.equal(cache_a.view(torch.uint8), cache_b.view(torch.uint8))
    times = {"kernel": [], "baseline": []}
    for name in ("kernel", "baseline", "baseline", "kernel"):
        fn = run_a if name == "kernel" else run_b
        times[name].append(timeit_graph([lambda i=i, fn=fn: fn(i) for i in range(copies)]) * 1e3)
    k, b = statistics.mean(times["kernel"]), statistics.mean(times["baseline"])
    moved = n * (64 * 512 * 3 + 512 * 3)
    print(
        f"rows={n:4d} copies={copies:3d}: kernel {k:7.2f} us (runs {times['kernel'][0]:.2f} / {times['kernel'][1]:.2f}), "
        f"baseline {b:7.2f} us (runs {times['baseline'][0]:.2f} / {times['baseline'][1]:.2f}), "
        f"{b / k:4.2f}x, kernel {moved / k / 1e3:5.0f} GB/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--rows", type=int, nargs="*", default=ROWS)
    args = parser.parse_args()
    with torch.inference_mode():
        for n in args.rows:
            bench(n, args.copies)
