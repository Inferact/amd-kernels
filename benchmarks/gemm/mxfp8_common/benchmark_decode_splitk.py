# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Interleaved A/B of the table's row against explicit decode / split-K decode rows on (16, 16) codes, per M: graph
replay over rotating weight copies > 400 MB (cold HBM), median of interleaved rounds. Every candidate is checked
first: the fp64 contract of reference.py, bitwise repeats and weight copies, and on order-sensitive data (per-32
scales over +-12 binades) whether it keeps the table row's bytes.

    python3 benchmark_decode_splitk.py --shape 4992,6144 --ms 96,176 \
        --cands "splitk:mb=1,waves=4,cols=1,loaders=4,ks=3,lw=2|decode:tm=48,tn=48,kw=3"

splitk keys: mb (m-blocks), waves (MFMA waves), cols (16-column tiles per wave), loaders, ks, lw, ring (0: all the
LDS holds); decode keys: tm, tn, kw, ks (split-K over workgroups).
"""
import argparse
import json
import statistics

import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.reference import COS_MIN, REL_L2_MAX, compare, mxfp8_gemm_reference, quantize_mxfp8

BF16 = torch.bfloat16
LLC_BYTES = 400 << 20


def _data(rows, cols, scale, seed, binades, spikes=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    e = torch.randint(-binades, binades + 1, (rows, cols // 32), device="cuda", generator=g).float()
    t = torch.randn(rows, cols, device="cuda", generator=g) * scale * torch.exp2(e).repeat_interleave(32, 1)
    if spikes:
        t[:, :: cols // 16] *= 8.0
    return t.to(BF16)


def config_of(spec, m):
    kind, args = spec.split(":")
    o = {key: int(v) for key, v in (kv.split("=") for kv in args.split(","))}
    if kind == "decode":
        return G._Config(o["tm"], o["tn"], 128, 0, 0, o.get("ks", 1), G._DECODE_KERNEL, o["kw"], 0)
    tile_m = 16 * -(-(-(-m // o["mb"])) // 16)
    return G._Config(tile_m, 16 * o["cols"] * o["waves"], 128, 0, 0, o["ks"], G._DECODE_SPLITK_KERNEL, o["waves"],
                     o["loaders"], o["lw"], o.get("ring", 0))


def calls_of(config, m, x, weights):
    q, s = G.mxfp8_e4m3_quantize_shuffled(x, max(16, config.tile_m))
    return [lambda w=w, ws=ws: G.mxfp8_gemm_shuffled(q, s, w, ws, BF16, m, config=config) for w, ws in weights], (q, s)


def build_graph(calls, inner=2):
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
    torch.cuda.synchronize()
    return g, len(calls) * inner


def time_graph(g, n_calls, replays):
    g.replay()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(replays):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / (replays * n_calls)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", required=True)
    p.add_argument("--ms", required=True)
    p.add_argument("--cands", required=True)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--replays", type=int, default=4)
    p.add_argument("--json")
    a = p.parse_args()
    n, k = map(int, a.shape.split(","))
    wq, we = quantize_mxfp8(_data(n, k, 0.05, n * 7 + k, 2))
    wwq, wwe = quantize_mxfp8(_data(n, k, 0.05, n * 3 + k, 12))
    wbytes = n * k * 33 // 32
    w0 = G.preshuffle_mxfp8_weight(wq.clone(), we.clone())
    weights = [w0] + [(w0[0].clone(), w0[1].clone()) for _ in range(max(2, -(-LLC_BYTES // wbytes)) - 1)]
    wide = [G.preshuffle_mxfp8_weight(wwq.clone(), wwe.clone())]
    res = {"N": n, "K": k, "m": {}}
    for m in map(int, a.ms.split(",")):
        x = _data(m, k, 1.0, 1000 + m, 2, spikes=True)
        ref = mxfp8_gemm_reference(*quantize_mxfp8(x), wq, we, accumulate=torch.float64)
        xw = _data(m, k, 1.0, 2000 + m, 12)
        refw = mxfp8_gemm_reference(*quantize_mxfp8(xw), wwq, wwe, accumulate=torch.float64)
        table = G._select_prequant_config(m, n, k, BF16)
        configs = {"table": table, **{spec: config_of(spec, m) for spec in a.cands.split("|")}}
        info, graphs, keep, table_wide = {}, {}, [], None
        for name, config in configs.items():
            calls, buf = calls_of(config, m, x, weights)
            wcalls, wbuf = calls_of(config, m, xw, wide)
            o1, o2, o3 = calls[0]().clone(), calls[1]().clone(), calls[0]().clone()
            ow = wcalls[0]().clone()
            table_wide = ow if name == "table" else table_wide
            std, wd = compare(o1, ref), compare(ow, refw)
            ok = (std["cos"] >= COS_MIN and std["rel_l2"] <= REL_L2_MAX and wd["cos"] >= COS_MIN
                  and torch.equal(o1, o2) and torch.equal(o1, o3) and bool(torch.isfinite(o1.float()).all()))
            info[name] = dict(config=str(config), std=std, wide=wd, ok=ok,
                              table_bytes=bool(torch.equal(ow, table_wide)))
            keep += [calls, buf, wcalls, wbuf]
            if ok:
                graphs[name] = build_graph(calls)
        samples = {key: [] for key in graphs}
        for _ in range(a.rounds):
            for key, (g, nc) in graphs.items():
                samples[key].append(time_graph(g, nc, a.replays))
        base = statistics.median(samples["table"])
        for key, v in samples.items():
            us = statistics.median(v)
            info[key].update(us=us, us_min=min(v), us_max=max(v))
            tb = wbytes / us / 1e6
            print(f"N{n} K{k} M{m} {key:52s} {us:7.2f} us {tb:4.2f} TB/s ({tb / 6.5:4.0%} of 6.5, {tb / 8:4.0%} of 8)"
                  f" vs table {us / base - 1:+6.1%} ok {info[key]['ok']} table bytes {info[key]['table_bytes']}",
                  flush=True)
        res["m"][m] = info
        del graphs, keep
        torch.cuda.empty_cache()
        if a.json:
            with open(a.json, "w") as f:
                json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
