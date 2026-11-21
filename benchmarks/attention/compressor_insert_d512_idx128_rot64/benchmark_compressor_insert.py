# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``compressor_insert`` against the vLLM / AITER chain it replaces (the baseline): the Triton
``fused_save_compress_norm`` and ``rope_quant_insert``, the wk GEMM and AITER's ``k_norm_rope_mxfp4_cache``, four
launches. Decode batches of 6-row requests padded to the row count; one HIP graph holds up to 100 launches, each with
its own raw rows, positions and slots (fewer when the inputs would pass 4 GiB). Rounds run A B B A (kernel, baseline,
baseline, kernel); each is the median of 5 x 10 replays, us per launch. The first launch of the kernel is checked
against the baseline's bytes (a few differ: the chain sums in the Triton order) before timing.
"""

import argparse
import statistics

import torch
from aiter.ops.triton.fusions.k_norm_rope_mxfp4_cache import k_norm_rope_mxfp4_cache
from amd_kernels.attention.compressor_insert_d512_idx128_rot64 import compressor_insert, shuffle_wk
from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import fused_save_compress_norm, rope_quant_insert

ROWS = [16, 48, 96, 192, 384, 768]
RING = 8
ENTRIES = {1: 128, 2: 64}
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


def make_inputs(ratio, n, copies, gen):
    reqs = n // 6
    e = ENTRIES[ratio]
    pages = 4096
    qsl = torch.arange(0, 6 * reqs + 1, 6, dtype=torch.int32, device="cuda")
    qsl = torch.cat([qsl, qsl[-1:]])[: reqs + 1]
    t2r = torch.zeros(n, dtype=torch.int32, device="cuda")
    t2r[: 6 * reqs] = torch.arange(reqs, device="cuda").repeat_interleave(6).int()
    out = []
    for _ in range(copies):
        p0 = torch.randint(1000, 100000, (reqs,), generator=gen, device="cuda").repeat_interleave(6)
        pos = torch.zeros(n, dtype=torch.int64, device="cuda")
        pos[: 6 * reqs] = p0 + torch.arange(6, device="cuda").repeat(reqs)
        ring_slots = torch.full((n,), -1, dtype=torch.int64, device="cuda")
        blk = torch.randperm(reqs, generator=gen, device="cuda") + 1
        ring_slots[: 6 * reqs] = blk.repeat_interleave(6) * RING + pos[: 6 * reqs] % RING

        def cache_slots():
            s = torch.full((n,), -1, dtype=torch.int64, device="cuda")
            s[: 6 * reqs] = torch.randperm((pages - 1) * e, generator=gen, device="cuda")[: 6 * reqs] + e
            return s

        raw = torch.randn((n, 512 * ratio), generator=gen, device="cuda")
        main_slots = cache_slots()
        out.append((raw, pos, ring_slots if ratio == 2 else main_slots, main_slots, cache_slots()))
    return qsl, t2r, out


def bench(ratio, n, wanted):
    gen = torch.Generator(device="cuda").manual_seed(n * 3 + ratio)
    copies = max(4, min(wanted, INPUT_BUDGET // (n * 512 * ratio * 4)))
    e = ENTRIES[ratio]
    qsl, t2r, inputs = make_inputs(ratio, n, copies, gen)
    ring = torch.randn((n // 6 + 2, RING, 1024), generator=gen, device="cuda") if ratio == 2 else None
    main_a = torch.zeros((4096, e, 512), dtype=torch.float8_e4m3fn, device="cuda")
    main_b = torch.zeros_like(main_a)
    idx_a = torch.zeros((4096, e * 68), dtype=torch.uint8, device="cuda")
    idx_b = torch.zeros_like(idx_a)
    cs_main = torch.randn((131072, 64), generator=gen, device="cuda")
    cs_idx = torch.randn((131072, 64), generator=gen, device="cuda")
    nw = (torch.rand(512, generator=gen, device="cuda") + 0.5).bfloat16()
    kw = (torch.rand(128, generator=gen, device="cuda") + 0.5).bfloat16()
    wk = (torch.randn((128, 512), generator=gen, device="cuda") * 0.05).bfloat16()
    wk_s = shuffle_wk(wk)
    scale = torch.ones(1, device="cuda")
    latent = torch.empty((n, 512), dtype=torch.bfloat16, device="cuda")

    def run_a(i):
        raw, pos, rs, ms, is_ = inputs[i]
        compressor_insert(raw, pos, ring, rs, qsl, t2r, nw, 1e-6, main_a, ms, cs_main, scale, wk_s, kw, 1e-6, idx_a, is_,
                          cs_idx, ratio)

    def run_b(i):
        raw, pos, rs, ms, is_ = inputs[i]
        fused_save_compress_norm(raw, pos, ring, rs, qsl, t2r, nw, 1e-6, ratio, latent)
        rope_quant_insert(latent, pos, cs_main, main_b, ms, ratio, fp8_scale=scale)
        k_norm_rope_mxfp4_cache(latent @ wk.T, pos, cs_idx, kw, 1e-6, idx_b, is_, ratio, shuffle=(32, 16, 2))

    run_a(0)
    run_b(0)
    torch.cuda.synchronize()
    diff = (idx_a != idx_b).sum().item(), (main_a.view(torch.uint8) != main_b.view(torch.uint8)).sum().item()
    times = {"kernel": [], "baseline": []}
    for name in ("kernel", "baseline", "baseline", "kernel"):
        fn = run_a if name == "kernel" else run_b
        times[name].append(timeit_graph([lambda i=i, fn=fn: fn(i) for i in range(copies)]) * 1e3)
    k, b = statistics.mean(times["kernel"]), statistics.mean(times["baseline"])
    print(
        f"ratio={ratio} rows={n:4d} copies={copies:3d}: kernel {k:7.2f} us (runs {times['kernel'][0]:.2f} / "
        f"{times['kernel'][1]:.2f}), baseline {b:7.2f} us (runs {times['baseline'][0]:.2f} / "
        f"{times['baseline'][1]:.2f}), {b / k:4.2f}x, bytes differing (indexer, main) {diff}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=100)
    parser.add_argument("--rows", type=int, nargs="*", default=ROWS)
    parser.add_argument("--ratios", type=int, nargs="*", default=[2, 1])
    args = parser.parse_args()
    with torch.inference_mode():
        for ratio in args.ratios:
            for n in args.rows:
                bench(ratio, n, args.copies)
