# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The row sweep behind ``gemm/mxfp8_n6144_k4096/host.py``: every candidate config of a family at each M, graph replay.

16 rotating random-weight copies (415 MB > MALL), 4 seeded inputs, a graph of 32 calls, alternated rounds, median us
per call; every candidate is checked against the reference first. ``decode``: the decode kernel on (16, 16) codes
(tiles 16..160 x 32..96, K-waves 1/2/4, owner-read on the 96-column two-K-wave tiles); ``decode_fq``: the bf16
activation quantized in the decode kernel; ``prefill``: row-major codes from a bf16 activation (quantizer timed) on
the LDS-staged, 4-wave, AITER and this folder's 256x192 / 128x192 tiles. ``--jobs`` compile-only processes build the
variants first (FlyDSL disk cache).

  python3 benchmarks/gemm/mxfp8_n6144_k4096/sweep_rows.py --family decode --ms 256,320,384
"""
import argparse
import os
import statistics
import subprocess
import sys

if os.environ.get("AKR_COMPILE_ONLY") == "1":
    import tests  # noqa: F401  (compile-only mode)
import torch
from amd_kernels.gemm.mxfp8_common import gemm as g, reference as ref
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode
from amd_kernels.gemm.mxfp8_n6144_k4096 import host as H

N, K = 6144, 4096
C = g._Config
COPIES = 16


def _decode_candidates(m):
    out = []
    for tm in sorted({t for t in (16, 32, 48, 64, 80, 96, 112, 128, 144, 160) if -(-m // t) <= 8 and t <= 16 * -(-m // 16)}):
        for ni in (2, 3, 4, 6):
            for kw in (1, 2, 4):
                out.append(("decode", C(tm, 16 * ni, 128, 0, 0, 1, g._DECODE_KERNEL, kw, 0), False))
                if ni == 6 and kw == 2:
                    out.append(("decode", C(tm, 96, 128, 0, 0, 1, g._DECODE_KERNEL, 2, 1), True))
    return out


def _decode_fq_candidates(m):
    tiles = [t for t in (4, 8, 16, 32) if t >= min(m, 16) or t == 16 * -(-m // 16)]
    return [("decode_fq", C(tm, 16 * ni, 128, 0, 0, 1, g._DECODE_FQ_KERNEL, kw, 0), False)
            for tm in tiles for ni in (2, 3, 4) for kw in (1, 2, 4, 8) if tm >= 16 or (32 // kw) % (16 // tm) == 0]


def _prefill_candidates(m):
    tile = lambda tm, tn, kernel, sk=1: C(tm, tn, 128, 1, 1, sk, kernel, 2, 2)  # noqa: E731
    out = [tile(256, 256, g._FOURWAVE_NT_KERNEL), tile(256, 256, g._FOURWAVE_NT_KERNEL, 0),
           tile(256, 128, g._FOURWAVE_NT_KERNEL), tile(128, 256, g._FOURWAVE_NT_KERNEL),
           tile(256, 192, H._PERSISTENT_KERNEL), C(128, 256, 128, 2, 4, 1), C(128, 128, 128, 2, 4, 1)]
    if m <= 2048:
        out.append(tile(256, 192, H._ONE_TILE_KERNEL))
    if m <= 1280:
        out += [tile(128, 192, H._TILE_128_KERNEL), C(128, 128, 128, 2, 4, 1, g._STAGED_KERNEL, 2, 4),
                C(64, 128, 128, 2, 4, 1, g._STAGED_KERNEL, 2, 2)]
    return [("prefill", c, False) for c in out]


FAMILIES = {"decode": _decode_candidates, "decode_fq": _decode_fq_candidates, "prefill": _prefill_candidates}


def _call(family, config, owner_read, xs, copies, m):
    if family == "decode_fq" or family == "prefill":
        return lambda j: g.mxfp8_linear(xs[j % 4], *copies[j % COPIES], config=config)
    qs = [g.mxfp8_e4m3_quantize_shuffled(x, config.tile_m) for x in xs]
    if not owner_read:
        return lambda j: g.mxfp8_gemm_shuffled(*qs[j % 4], *copies[j % COPIES], torch.bfloat16, m, config=config)
    from amd_kernels.common.tensor_shim import _run_compiled
    exe = compile_mxfp8_gemm_decode(N=N, K=K, RT=config.tile_m // 16, NI=6, KW=2, OWNER_READ=True)
    u8 = lambda t: t.view(torch.uint8).view(-1)  # noqa: E731
    n_mb = -(-m // config.tile_m)

    def run(j):
        (a, sa), (w, sw) = qs[j % 4], copies[j % COPIES]
        out = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")
        _run_compiled(exe, u8(a), u8(w), u8(out), u8(sa), u8(sw), m, sa.numel(), n_mb, n_mb * (N // 96),
                      torch.cuda.current_stream())
        return out
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=FAMILIES, required=True)
    ap.add_argument("--ms", required=True)
    ap.add_argument("--rounds", type=int, default=9)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=-1)
    args = ap.parse_args()
    compile_only = args.shard >= 0
    if not compile_only and args.jobs > 1:
        env = dict(os.environ, AKR_COMPILE_ONLY="1")
        for p in [subprocess.Popen([sys.executable] + sys.argv + ["--shard", str(i)], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for i in range(args.jobs)]:
            p.wait()
    torch.manual_seed(0)
    raw = ref.quantize_mxfp8(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.05)
    copies = [g.preshuffle_mxfp8_weight(raw[0].clone(), raw[1].clone()) for _ in range(1 if compile_only else COPIES)]
    copies = copies * (COPIES // len(copies))
    for m in map(int, args.ms.split(",")):
        xs = [torch.randn(m, K, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        expected = ref.mxfp8_linear_reference(xs[0], *raw)
        calls = {}
        for i, (family, config, owner_read) in enumerate(FAMILIES[args.family](m)):
            if compile_only and i % args.jobs != args.shard:
                continue
            label = f"{config.kernel} {config.tile_m}x{config.tile_n} kw{config.waves_m} sk{config.split_k}" + (
                " sb owner-read" if owner_read else "")
            try:
                call = _call(args.family, config, owner_read, xs, copies, m)
                got = call(0)
                torch.cuda.synchronize()
            except Exception as e:  # an illegal combination (registers, LDS)
                print(f"M {m} {label}: {type(e).__name__}", flush=True)
                continue
            if compile_only:
                continue
            metrics = ref.compare(got, expected)
            if metrics["cos"] >= ref.COS_MIN and metrics["rel_l2"] <= ref.REL_L2_MAX:
                calls[label] = call
        if compile_only:
            continue
        graphs = {}
        for label, call in calls.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for j in range(2 * COPIES):
                    call(j)
            graph.replay()
            graphs[label] = graph
        torch.cuda.synchronize()
        times = {label: [] for label in graphs}
        for r in range(args.rounds):
            for label in (list(graphs) if r % 2 == 0 else list(graphs)[::-1]):
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(10):
                    graphs[label].replay()
                stop.record()
                stop.synchronize()
                times[label].append(start.elapsed_time(stop) * 1000 / (10 * 2 * COPIES))
        for label, us in sorted(((k, statistics.median(v)) for k, v in times.items()), key=lambda t: t[1])[:6]:
            print(f"M {m:6d} {us:9.2f} us  {label}", flush=True)
        del graphs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
