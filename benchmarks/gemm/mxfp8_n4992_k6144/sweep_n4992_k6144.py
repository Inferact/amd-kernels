# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The sweeps behind ``gemm/mxfp8_n4992_k6144/host.py`` (random MXFP8 weights, 14 cold copies > 400 MB, graph
replay; every candidate is checked against the reference first, failures are listed and skipped).

  --decode M,..     decode rows: row tile x column tile x K-waves x K-split (GEMM only, (16, 16) codes)
  --chain M,..      the same rows + the TP2 norm / RoPE / NVFP4 insert (split-K rows: partials + reduce-insert)
  --prefill M,..    persistent tiles 256/192/128 x 192/128 (row-major codes), per-round us
  --split M,..      host.persistent_plan's row split against its single-launch tile (bitwise equal)

  python3 benchmarks/gemm/mxfp8_n4992_k6144/sweep_n4992_k6144.py --decode 16,64,256 --out sweep.json
"""

import argparse
import json
import statistics

import numpy as np
import torch

from amd_kernels.gemm.mxfp8_common import gemm as g
from amd_kernels.gemm.mxfp8_common.reference import compare, mxfp8_gemm_reference, quantize_mxfp8
from amd_kernels.gemm.mxfp8_n4992_k6144 import host as H

N, K = H.N, H.K
LLC_BYTES = 420 << 20
TILES = ((256, 192), (256, 128), (192, 192), (192, 128), (128, 192), (128, 128))


def timeit_graph(calls, inner=2, replays=10, rounds=5):
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(3):
        graph.replay()
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events) * 1e3 / (replays * len(calls) * inner)


class Bench:
    def __init__(self):
        torch.manual_seed(5)
        self.weights = []
        for _ in range(-(-LLC_BYTES // (N * K * 33 // 32))):
            codes, scales = quantize_mxfp8(torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.5)
            self.weights.append(g.preshuffle_mxfp8_weight(codes, scales))
            if len(self.weights) == 1:
                self.raw = (codes, scales)
        self.m = None

    def inputs(self, m):
        if self.m != m:
            torch.manual_seed(m)
            self.x = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
            self.ref = mxfp8_gemm_reference(*quantize_mxfp8(self.x), *self.raw)
            self.m = m
        return self.x

    def gemm(self, m, config, chain=None):
        x = self.inputs(m)
        out = torch.empty((m, N), device="cuda", dtype=torch.bfloat16)
        if config.kernel == g._DECODE_KERNEL:
            a, asc = g.mxfp8_e4m3_quantize_shuffled(x, max(16, config.tile_m))

            defer = chain is not None and config.split_k > 1

            def call(w, s):
                part = g._launch_decode(config, a, asc, w, s, out, defer_reduce=defer)
                if chain is not None:
                    chain(out, part)
        else:
            a, asc = g.mxfp8_e4m3_quantize_aiter(x)
            launch = H.LAUNCHERS[config.kernel]

            def call(w, s):
                launch(config, a, asc, w, s, out)
        if chain is None:
            call(*self.weights[0])
            torch.cuda.synchronize()
            metrics = compare(out, self.ref)
            if not (metrics["cos"] >= 0.99999 and metrics["rel_l2"] <= 1e-3):
                raise AssertionError(f"mismatch {metrics}")
        return timeit_graph([lambda w=w, s=s: call(w, s) for w, s in self.weights])


def insert_chain(m):
    """The TP2 insert consumer (heads 32 / 2 / 2) of ``m`` decode tokens (8 per request)."""
    from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
        norm_rope_nvfp4_insert, splitk_reduce_norm_rope_nvfp4_insert)

    rng = np.random.default_rng(m)
    pos, slots, total = [], [], 0
    for length in [8] * (m // 8) + ([m % 8] if m % 8 else []):
        start = int(rng.integers(0, 131072 - length))
        p = np.arange(start, start + length)
        pos.extend(p)
        slots.extend((total + p // 128 - start // 128) * 128 + p % 128)
        total += (start + length - 1) // 128 - start // 128 + 1
    s = torch.tensor(slots, device="cuda")
    perm = torch.tensor(rng.permutation(total), device="cuda")
    s = perm[s // 128] * 128 + s % 128
    f = torch.arange(131072, dtype=torch.float64)[:, None] / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    one = torch.tensor(1.0, device="cuda")
    ins = ([(torch.rand(128, device="cuda") + 0.5).bfloat16() for _ in range(4)]
           + [torch.cat([f.cos(), f.sin()], 1).to(torch.bfloat16).cuda().contiguous(),
              torch.tensor(pos, device="cuda"), s, s,
              torch.zeros(total, 4, 128, 72, dtype=torch.uint8, device="cuda"),
              torch.zeros(total, 128, 72, dtype=torch.uint8, device="cuda"), one, one,
              torch.empty(m, 4096, dtype=torch.bfloat16, device="cuda"),
              torch.empty(m, 256, dtype=torch.float8_e4m3fn, device="cuda")])
    kw = dict(num_heads=32, num_kv_heads=2, num_index_heads=2, eps=1e-6)

    def chain(out, part):
        if isinstance(part, g.SkPartials):
            splitk_reduce_norm_rope_nvfp4_insert(part.tmp, part.split_k, *ins, **kw)
        else:
            norm_rope_nvfp4_insert(out, *ins, **kw)
    return chain


def decode_candidates(m, nis=(2, 3, 4, 6, 8), kws=(1, 2, 3, 4), kss=(1, 2, 3), workgroups=(150, 1400)):
    for n_mb in range(1, 17):
        rt = -(-m // (16 * n_mb))
        if rt > 16 or (n_mb > 1 and -(-m // (16 * rt)) != n_mb):
            continue
        for ni in nis:
            for kw in kws:
                for ks in kss:
                    wgs = ks * n_mb * (N // 16 // ni)
                    if 48 % (ks * kw) or (48 // (ks * kw)) % 2 or not workgroups[0] <= wgs <= workgroups[1]:
                        continue
                    if rt * ni * 48 // (ks * kw) > 640:  # unrolled MFMAs per wave: larger ones compile for minutes
                        continue
                    yield g._Config(16 * rt, 16 * ni, 128, 0, 0, ks, g._DECODE_KERNEL, kw, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decode", default="")
    ap.add_argument("--chain", default="")
    ap.add_argument("--prefill", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--ni", default="2,3,4,6,8", help="16-column tiles per workgroup")
    ap.add_argument("--kw", default="1,2,3,4", help="K-waves")
    ap.add_argument("--ks", default="1,2,3", help="K splits")
    ap.add_argument("--workgroups", default="150,1400")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--out")
    args = ap.parse_args()
    bench, result = Bench(), {}
    jobs = [("decode", m) for m in args.decode.split(",") if m] + [("chain", m) for m in args.chain.split(",") if m]
    jobs += [("prefill", m) for m in args.prefill.split(",") if m]
    for m in (int(v) for v in args.split.split(",") if v):
        plan = H.persistent_plan(m)
        if len(plan) == 1:
            print(f"split M {m}: one launch {plan[0][1]}", flush=True)
            continue
        single = bench.gemm(m, H._persistent(*H.persistent_tile(m)))
        split = bench.gemm(m, H._persistent(*plan[0][1], H._PERSISTENT_SPLIT_KERNEL))
        print(f"split M {m}: {plan} {split:.1f} us vs one launch {H.persistent_tile(m)} {single:.1f} us "
              f"({100 * (split / single - 1):+.1f}%)", flush=True)
        result[f"split:{m}"] = [split, single]
    for kind, m in jobs:
        m = int(m)
        if kind == "prefill":
            configs = [H._persistent(*t) for t in TILES]
        else:
            ints = lambda v: tuple(int(x) for x in v.split(","))  # noqa: E731
            configs = list(decode_candidates(m, ints(args.ni), ints(args.kw), ints(args.ks), ints(args.workgroups)))
        chain = insert_chain(m) if kind == "chain" else None
        rows = []
        for config in configs:
            try:
                if chain is not None:
                    bench.gemm(m, config)  # the reference check
                us = bench.gemm(m, config, chain)
            except Exception as e:  # noqa: BLE001
                print(f"  fail {config}: {str(e)[:120]}", flush=True)
                continue
            rounds = H._rounds(m, (config.tile_m, config.tile_n)) if kind == "prefill" else 0
            rows.append((us, config.tile_m, config.tile_n, config.waves_m, config.split_k, rounds))
        rows.sort()
        print(f"{kind} M {m}", flush=True)
        for us, tm, tn, kw, ks, rounds in rows[: args.top]:
            extra = f"{rounds} rounds, {us / rounds:.1f} us/round" if rounds else f"kw{kw} ks{ks}"
            print(f"  {us:8.2f} us  {tm}x{tn} {extra}", flush=True)
        result[f"{kind}:{m}"] = rows
        if args.out:
            with open(args.out, "w") as f:
                json.dump(result, f)


if __name__ == "__main__":
    main()
