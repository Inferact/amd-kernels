# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay sweep of the table rows of one (N, K): every candidate configuration at every M, weights HBM-cold.

A candidate is timed through the bf16 entry (``mxfp8_linear`` with an explicit config: the activation quantizer the
row needs plus the GEMM, or the in-kernel quantizer of ``decode_fq``), or with ``--entry codes`` on codes made
before timing (the producer path: (16, 16) codes for decode rows, row-major otherwise). Each candidate is checked
against ``reference.py`` on the first weight copy; the first pass times every candidate briefly, the second
re-times the best ``--refine`` per M. ``--table`` adds the table's own row, ``--baseline`` vLLM's Triton
``tl.dot_scaled`` linear (needs vLLM on the path).

    python3 benchmarks/gemm/mxfp8_common/sweep_rows.py --shape 3072,6144 --family decode --ms 1,8,64,256
"""

import argparse
import dataclasses
import functools
import json
import statistics

import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_common.gemm import _Config
from amd_kernels.gemm.mxfp8_common.reference import COS_MIN, REL_L2_MAX, compare, mxfp8_linear_reference, quantize_mxfp8
from benchmarks.gemm.mxfp8_common.benchmark_gemm import LLC_TARGET_BYTES, MAX_SLOTS, timeit_graph


def _ks_kw_pairs(k):
    steps = k // 128
    return [(ks, kw) for ks in (1, 2, 3, 4, 6, 8) for kw in (1, 2, 3, 4)
            if steps % (ks * kw) == 0 and (steps // (ks * kw)) % 2 == 0]


def decode_candidates(m, n, k, w_cms=(None, 0)):
    """Decode-kernel rows: the smallest row tile per m-block count (1-4 blocks; 5-8 with cached W, one or two
    K-waves), column tiles 16..128, K-waves 1-4, split-K where one split leaves the grid under 256 workgroups."""
    out = []
    seen_blocks = set()
    for rt in range(1, 13):
        blocks = -(-m // (16 * rt))
        if blocks > 8 or blocks in seen_blocks:
            continue
        seen_blocks.add(blocks)
        for ni in (1, 2, 3, 4, 5, 6, 8):
            if n % (16 * ni) or rt * ni > 64:
                continue
            base = blocks * n // (16 * ni)
            for ks, kw in _ks_kw_pairs(k):
                if ks > 1 and base >= 256 or ks * base > 8192 or ks * base < 64:
                    continue
                if (kw - 1) * rt * ni * 1024 > 160 * 1024:
                    continue
                if blocks > 4 and (kw > 2 or ks > 1 or 0 not in w_cms):
                    continue
                for w_cm in (w_cms if blocks <= 4 else (0,)) if blocks > 1 else (None,):
                    out.append(_Config(16 * rt, 16 * ni, 128, 0, 0, ks, "decode", kw, 0, w_cm=w_cm))
    return out


def decode_fq_candidates(m, n, k):
    """In-kernel-quantizer rows: tile 4 / 8 (folded) or 16 / 32 rows, one m-block."""
    out = []
    tiles = [t for t in (4, 8, 16, 32, 64) if t >= m][:2]
    for tile in tiles:
        rt = max(1, tile // 16)
        for ni in (1, 2, 3, 4, 6):
            if n % (16 * ni):
                continue
            base = n // (16 * ni)
            for ks, kw in _ks_kw_pairs(k):
                sw = k // 128 // (ks * kw)
                fold = 16 // tile if tile in (4, 8) else 1
                if sw % fold or (ks > 1 and base >= 256) or ks * base < 64:
                    continue
                out.append(_Config(tile, 16 * ni, 128, 0, 0, ks, "decode_fq", kw, 0))
    return out


def aiter_candidates(m, n, k):
    out = []
    for tm, tn, tk in ((32, 32, 256), (32, 64, 128), (32, 64, 256), (32, 128, 256), (64, 64, 256), (64, 128, 256),
                       (128, 64, 256), (128, 128, 128), (128, 128, 256), (128, 256, 128)):
        if n % tn or (tm >= 128 and m <= 64):
            continue
        tiles = -(-m // tm) * (n // tn)
        for wpe, xcd in ((0, 0), (2, 4)):
            for sk in (1, 2, 3, 4, 6, 8):
                if k % (256 * sk) or (sk > 1 and tiles >= 256) or sk * tiles > 8192:
                    continue
                out.append(_Config(tm, tn, tk, wpe, xcd, sk))
    return out


def fourwave_candidates(m, n, k):
    if n % 256:
        return []
    out = []
    for tm, tn in ((256, 256), (128, 256), (256, 128), (128, 128)):
        for kernel in ("4wave", "4wave_nt"):
            for sk in ((0, 1) if (tm, tn) == (256, 256) else (1,)):
                out.append(_Config(tm, tn, 128, 1, 1, sk, kernel, 2, 2))
    return out


def staged_candidates(m, n, k):
    out = []
    for tm, tn, wn in ((128, 128, 4), (128, 256, 4), (256, 128, 4), (64, 128, 2), (128, 64, 2)):
        if n % tn or tm > 64 and m < 512:
            continue
        for sk in (1, 2):
            if k % (256 * sk):
                continue
            out.append(_Config(tm, tn, 128, 2, 4, sk, "staged_ab", 2, wn))
    return out


def tall_candidates(m, n, k):
    """Decode rows of 13-16 row tiles (one m-block reads W once at M <= 256)."""
    out = []
    for rt in range(13, 17):
        for ni in (1, 2, 3, 4):
            if n % (16 * ni) or -(-m // (16 * rt)) > 2:
                continue
            for ks, kw in _ks_kw_pairs(k):
                if ks > 2 or (kw - 1) * rt * ni * 1024 > 160 * 1024:
                    continue
                for w_cm in (None, 0):
                    out.append(_Config(16 * rt, 16 * ni, 128, 0, 0, ks, "decode", kw, 0, w_cm=w_cm))
    return out


def knob_candidates(m, n, k):
    """The table's decode row at ``m`` with scheduling barriers on / off and shallower lookaheads."""
    row = G._select_config(m, n, k)
    if row.kernel not in ("decode", "decode_fq"):
        return []
    out = []
    for waves_n in (0, 1):
        for la in ((0, 0), (1, 1), (1, 2), (2, 2), (2, 4), (4, 4), (4, 8), (6, 6), (8, 8), (8, 16)):
            out.append(dataclasses.replace(row, waves_n=waves_n, la_w=la[0], la_a=la[1]))
    return out


FAMILIES = {
    "knobs": knob_candidates,
    "tall": tall_candidates,
    "decode": lambda m, n, k: decode_fq_candidates(m, n, k) * (m <= 64) + decode_candidates(m, n, k),
    "decode_nocm": lambda m, n, k: decode_fq_candidates(m, n, k) * (m <= 64) + decode_candidates(m, n, k, (None,)),
    "aiter": aiter_candidates,
    "all": lambda m, n, k: (decode_fq_candidates(m, n, k) * (m <= 64) + decode_candidates(m, n, k)
                            + aiter_candidates(m, n, k) * (m >= 64)
                            + (fourwave_candidates(m, n, k) + staged_candidates(m, n, k)) * (m >= 256)),
    "prefill": lambda m, n, k: fourwave_candidates(m, n, k) + aiter_candidates(m, n, k) + staged_candidates(m, n, k),
}


def _key(cfg):
    if cfg is None:
        return "table"
    if isinstance(cfg, str):
        return cfg
    s = f"{cfg.kernel} {cfg.tile_m}x{cfg.tile_n}"
    if cfg.kernel.startswith("decode"):
        s += f" kw{cfg.waves_m} ks{cfg.split_k}" + ("" if cfg.w_cm is None else f" cm{cfg.w_cm}")
        s += (f" la{cfg.la_w}x{cfg.la_a}" if cfg.la_w else "") + (" sb" if cfg.waves_n else "")
    elif cfg.kernel.startswith("4wave"):
        s += f" sk{cfg.split_k} xcd{cfg.xcd_swizzle}"
    else:
        s += f" k{cfg.tile_k} wpe{cfg.waves_per_eu} xcd{cfg.xcd_swizzle} sk{cfg.split_k}"
        if cfg.kernel == "staged_ab":
            s += f" wn{cfg.waves_n}"
    return s


@functools.cache
def _decode_launcher_with(opts, n, k, config, a_bf16):
    """``gemm._decode_launcher`` with kernel options the table rows do not carry (``GROUPED``, ...)."""
    from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_decode import compile_mxfp8_gemm_decode

    return compile_mxfp8_gemm_decode(
        N=n, K=k, RT=max(1, config.tile_m // 16), NI=config.tile_n // 16, KW=config.waves_m, KS=config.split_k,
        LA=(config.la_w, config.la_a) if config.la_w else None, sched_barriers=bool(config.waves_n), A_BF16=a_bf16,
        FOLD=G._decode_fold(config) if a_bf16 else 1, W_CM=config.w_cm, **dict(opts),
    )


class Bench:
    def __init__(self, n, k, baseline):
        self.n, self.k = n, k
        torch.manual_seed(n + k)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
        self.codes, self.exps = quantize_mxfp8(w)
        sw, ss = G.preshuffle_mxfp8_weight(self.codes, self.exps)
        wbytes = n * k * 33 // 32
        self.slots = min(MAX_SLOTS, max(2, -(-LLC_TARGET_BYTES // wbytes)))
        self.weights = [(sw.clone(), ss.clone()) for _ in range(self.slots)]
        self.rm = [(self.codes.clone(), self.exps.clone()) for _ in range(self.slots)] if baseline else None
        self.x = None
        self.ref = None

    def set_m(self, m):
        torch.manual_seed(m)
        self.x = torch.randn(m, self.k, device="cuda", dtype=torch.bfloat16)
        self.ref = mxfp8_linear_reference(self.x, self.codes, self.exps)

    def calls(self, cfg, entry):
        x, m = self.x, self.x.shape[0]
        if cfg == "baseline":
            from vllm.model_executor.kernels.linear.mxfp8.rocm_native import _mxfp8_dot_scaled_linear

            return [lambda w=w, s=s: _mxfp8_dot_scaled_linear(x, w, s) for w, s in self.rm]
        if entry == "linear":
            return [lambda w=w, s=s: G.mxfp8_linear(x, w, s, torch.bfloat16, config=cfg) for w, s in self.weights]
        if cfg is None:  # the table's row on the codes its producer writes
            tile = G.prequant_code_tile_rows(m, self.n, self.k)
            q, s = G.mxfp8_e4m3_quantize_shuffled(x, tile) if tile else G.mxfp8_e4m3_quantize_aiter(x)
            return [lambda w=w, ws=ws: G.mxfp8_gemm_prequant(q, s, w, ws, torch.bfloat16, m) for w, ws in self.weights]
        if cfg.kernel.startswith("decode"):
            c = dataclasses.replace(cfg, kernel="decode", tile_m=max(16, cfg.tile_m), min_m=0)
            q, s = G.mxfp8_e4m3_quantize_shuffled(x, c.tile_m)
            return [lambda w=w, ws=ws: G.mxfp8_gemm_shuffled(q, s, w, ws, torch.bfloat16, m, config=c)
                    for w, ws in self.weights]
        q, s = G.mxfp8_e4m3_quantize_aiter(x)
        return [lambda w=w, ws=ws: G.mxfp8_gemm(q, s, w, ws, torch.bfloat16, True, config=cfg)
                for w, ws in self.weights]

    def time(self, cfg, entry, fast, opts=()):
        if opts:
            original = G._decode_launcher
            G._decode_launcher = functools.partial(_decode_launcher_with, opts)
            try:
                return self.time(cfg, entry, fast)
            finally:
                G._decode_launcher = original
        calls = self.calls(cfg, entry)
        got = calls[0]()
        torch.cuda.synchronize()
        met = compare(got, self.ref)
        if not (met["cos"] >= COS_MIN and met["rel_l2"] <= REL_L2_MAX):
            raise AssertionError(f"{_key(cfg)}: {met}")
        m = self.x.shape[0]
        flops = 2 * m * self.n * self.k
        work = max(flops * len(calls), 1)
        replays = max(2, min(20, int(2e10 // work))) if fast else max(5, min(50, int(1e11 // work)))
        return timeit_graph(calls, inner=2, replays=replays, rounds=3 if fast else 7, warmup=2 if fast else 5) * 1e3


def roofline_us(m, n, k, bw=6.5e12, fp8=5e15):
    bytes_ = n * k * 33 / 32 + m * k * 2 + m * n * 2
    return max(bytes_ / bw, 2 * m * n * k / fp8) * 1e6


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", required=True)
    p.add_argument("--ms", required=True)
    p.add_argument("--family", default="decode", choices=[*FAMILIES, "none"])
    p.add_argument("--entry", default="linear", choices=("linear", "codes"))
    p.add_argument("--table", action="store_true")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--refine", type=int, default=6)
    p.add_argument("--repeat", type=int, default=1, help="interleaved repeats of the final timings")
    p.add_argument("--grouped", action="store_true", help="also the best decode rows with the grouped XCD order")
    p.add_argument("--extra", default="", help="';'-separated _Config(...) argument lists timed at every M")
    p.add_argument("--json")
    args = p.parse_args()
    n, k = map(int, args.shape.split(","))
    ms = [int(v) for v in args.ms.split(",")]
    bench = Bench(n, k, args.baseline)
    extra = [eval(f"_Config({e})") for e in args.extra.split(";") if e.strip()]
    results = {}
    for m in ms:
        bench.set_m(m)
        cands = (FAMILIES[args.family](m, n, k) if args.family != "none" else []) + extra
        times = {}
        for cfg in cands:
            try:
                times[cfg] = bench.time(cfg, args.entry, fast=True)
            except Exception as e:  # noqa: BLE001 - a candidate the kernel rejects
                print(f"  M {m} {_key(cfg)}: {type(e).__name__} {str(e)[:120]}", flush=True)
        best = sorted(times, key=times.get)[: args.refine]
        entries = [(_key(cfg), cfg, ()) for cfg in best + ([None] if args.table else [])
                   + (["baseline"] if args.baseline else [])]
        if args.grouped:
            entries += [(_key(cfg) + " g4", cfg, (("GROUPED", True),)) for cfg in best
                        if cfg.kernel == "decode" and cfg.split_k == 1 and -(-m // cfg.tile_m) > 1]
        samples = {}
        for _ in range(args.repeat):  # interleaved, the median per entry
            for key, cfg, opts in entries:
                try:
                    samples.setdefault(key, []).append(bench.time(cfg, args.entry, False, opts))
                except Exception as e:  # noqa: BLE001
                    print(f"  M {m} {key}: {type(e).__name__} {str(e)[:200]}", flush=True)
        cfgs = {key: (G._select_config(m, n, k) if cfg is None else cfg) for key, cfg, _ in entries}
        final = {key: (statistics.median(v), cfgs[key]) for key, v in samples.items()}
        order = sorted((v[0], key) for key, v in final.items() if v is not None)
        roof = roofline_us(m, n, k)
        line = "  ".join(f"{key} {us:.2f}" for us, key in order)
        print(f"N{n} K{k} M {m} ({len(cands)} cands, roof {roof:.2f}): {line}", flush=True)
        results[m] = {key: {"us": v[0], "cfg": None if not isinstance(v[1], _Config) else v[1].__dict__}
                      for key, v in final.items() if v is not None}
        results[m]["roofline_us"] = roof
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"n": n, "k": k, "entry": args.entry, "results": results}, f, indent=1)


if __name__ == "__main__":
    main()
