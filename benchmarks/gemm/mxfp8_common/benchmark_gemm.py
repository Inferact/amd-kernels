# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of explicit MXFP8 GEMM schedules on gfx950, weights HBM-cold.

Timing (``timeit_graph``): one HIP graph cycles over rotating weight copies whose footprint
exceeds the 256 MB MALL, so every call reads its weight from HBM, as in serving. The
activation is quantized before timing and shared by all slots (producer-warm); the
timed launch is the GEMM (with its split-K / tail fixup), plus the in-kernel quantizer
for ``decode_fq``; ``gateup`` times the fused gate_up + SwiGLU-OAI + MXFP8 quant row.
Every slot is checked against ``mxfp8/reference.py`` first.
"""

import argparse
import statistics

import torch
from amd_kernels.gemm.mxfp8_common import gemm
from amd_kernels.gemm.mxfp8_common.gemm import (
    _dummy_i32,
    _launch_decode,
    _launch_fourwave,
    _launch_tiled,
    mxfp8_e4m3_quantize_flydsl,
    mxfp8_e4m3_quantize_shuffled,
    preshuffle_mxfp8_weight,
)
from amd_kernels.gemm.mxfp8_common.reference import (
    COS_MIN,
    REL_L2_MAX,
    compare,
    mxfp8_linear_reference,
    quantize_mxfp8,
)
from tests.gemm.mxfp8_common.test_gemm import CASES

LLC_TARGET_BYTES = 400 * 1024**2  # above the 256 MB MALL
MALL_BYTES = 256 * 1024**2
MAX_SLOTS = 256  # a graph of thousands of launches aborts in HIP (a 135 KB weight wanted 3100)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=[case[0] for case in CASES], required=True
    )
    parser.add_argument(
        "--dbo", action="store_true", help="the 4-wave rows' plain grid of a DBO microbatch"
    )
    args = parser.parse_args()
    if args.dbo:
        gemm.DBO = True
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        raise RuntimeError("requires gfx950")
    name, m, n, k, config = next(case for case in CASES if case[0] == args.variant)
    kernel = config.kernel
    torch.manual_seed(17)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.5
    weight_bytes = n * k + n * k // 32
    slots = min(MAX_SLOTS, max(2, -(-LLC_TARGET_BYTES // weight_bytes)))
    footprint = slots * weight_bytes
    weights = []
    for _ in range(slots):
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.5
        codes, exponents = quantize_mxfp8(w)
        weights.append((codes, exponents, *preshuffle_mxfp8_weight(codes, exponents)))
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    a_bytes = m * k + m * k // 32
    if kernel == "decode_fq":
        launch = lambda w, s: _launch_decode(config, x, _dummy_i32(x.device), w, s, out, True)
        a_bytes = 2 * m * k
    elif kernel == "decode":
        act, act_scale = mxfp8_e4m3_quantize_shuffled(x, config.tile_m)
        launch = lambda w, s: _launch_decode(config, act, act_scale, w, s, out)
    elif kernel in ("4wave", "4wave_nt"):
        act, act_scale = mxfp8_e4m3_quantize_flydsl(x)
        launch = lambda w, s: _launch_fourwave(config, act, act_scale, w, s, out)
    else:
        act, act_scale = mxfp8_e4m3_quantize_flydsl(x)
        launch = lambda w, s: _launch_tiled(config, act, act_scale, w, s, out)

    calls = [lambda w=w, s=s: launch(w, s) for _, _, w, s in weights]
    for call, (codes, exponents, _, _) in zip(calls, weights):
        call()
        torch.cuda.synchronize()
        metrics = compare(out, mxfp8_linear_reference(x, codes, exponents))
        ok = metrics["cos"] >= COS_MIN and metrics["rel_l2"] <= REL_L2_MAX
        assert ok, metrics

    us = timeit_graph(calls) * 1e3
    out_bytes = 2 * m * n
    moved = a_bytes + weight_bytes + out_bytes
    print(
        f"{name}{' (DBO plain grid)' if gemm.DBO else ''}: M={m} N={n} K={k}, "
        "BF16 output, "
        f"{slots} weight slots "
        f"({footprint / 2**20:.0f} MB{', cache-warm' if footprint < MALL_BYTES else ''}), "
        f"{us:.2f} us, {2 * m * n * k / us / 1e6:.1f} TFLOPS, {moved / us / 1e3:.0f} GB/s"
    )


if __name__ == "__main__":
    main()
