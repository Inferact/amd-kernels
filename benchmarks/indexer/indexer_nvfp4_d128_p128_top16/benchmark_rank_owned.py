# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four-GPU timing of the rank-owned decode indexer (``split4``: rank r holds and
scores the blocks b % 4 == r), through the host calls the served model makes.

One process per GPU (gloo group, HIP IPC buffers as ``init_rank_owned`` makes
them); rank r runs on card r of ``HIP_VISIBLE_DEVICES`` (four cards).

- decode unit (graph replay, as the FULL decode graphs captured at
  max_model_len 1M): ``get_rank_owned(0).score_topk`` (one launch; a package
  without it: ``score_push`` then ``topk.index_topk_decode_indices(...,
  rank_owned_order=True)``), ``--ql`` rows per
  request (default 8, DSpark 7), M / ql requests of context C each (row j sees
  C - ql + 1 + j tokens); ``--reqs N,..``: N requests of the contexts
  ``F[str(N)]`` of ``--lens-json F`` instead, a JSON file {"N": [context length
  of request i of N, ...]} (e.g. the (i + 0.5) / N quantiles of a workload's
  request context lengths); ``--reqs`` without ``--lens-json``:
  ``synthetic_lens``, request i of N at the (i + 0.5) / N quantile of a
  log-uniform 32K-192K distribution.

Inputs: every rank derives the same global data and builds only its own pages:
global page p = 4 s + c lives at slot s of rank c, filled from a generator
seeded by (c, chunk) with Gaussian keys quantized as the model does
(``reference.nvfp4_index_pages``); the block tables keep p % 4 == b % 4, each
request's blocks on distinct random pages of its input's region of the pool.
Index-q rows: Gaussian, e4m3, ``np.random.default_rng(1000 * M + i)``, rank r
takes head r. The pool holds >= 3 of the largest inputs and >= 768 MB per rank;
a graph holds at least 100 calls and enough calls that an input's pages come
back after >= 384 MB of others, including replay wraparound.

Timing: 2 warm-up replays, then ``--rounds`` rounds of 10 replays; each round
runs every side back to back (order alternated), after a host barrier and one
small rank-owned call as a device barrier; a sample = the max over the four
ranks of that rank's event time per call. ``--ref DIR``: a frozen copy of this
package (its directory under any name, imported as its own module) timed as a
second side on the same inputs; outputs of the two sides compared on the first
and last input (equal ids per row, else the count of rows that differ).
``--profile``: the per-kernel times over at most 16 calls per side (torch.profiler),
the slowest rank's. Hosts synchronize after profiler initialization.
``--arms name:tk=R,grid=G;..``: more decode sides of this checkout on the same
buffers (``tk``: ``decode_score.TK_MAX_ROWS`` at capture, 0 = two launches;
``grid``: the scorer's grid), timed with the others; each round's per-rank
times go to the JSON record (``ranks``).

    HIP_VISIBLE_DEVICES=0,1,2,3 python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_rank_owned.py \
        --decode 8,256 --ctx 32768,131072 --ref /path/to/frozen/indexer_nvfp4_d128_p128_top16
"""

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import socket
import statistics
import sys
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist

BLK, D, TOPK = 128, 128, 16
PAGE_BYTES = BLK * 72
QL = 8  # decode rows per request (--ql; DSpark 7: 8)
INIT, LOCAL = 0, 1
MAX_MODEL_LEN = 1 << 20
MAX_REQS = 256
SETS = 2
MIN_POOL_BYTES = 768 << 20
REUSE_BYTES = 384 << 20
WORLD = 4

DECODE_M = (8, 16, 32, 128, 256, 512)
DECODE_CTX = (32768, 131072, 524288)
SYNTHETIC_RANGE = (32768, 196608)  # the log-uniform context range of --reqs without --lens-json


def synthetic_lens(n, lo=SYNTHETIC_RANGE[0], hi=SYNTHETIC_RANGE[1]):
    """Request i of n at the (i + 0.5) / n quantile of a log-uniform [lo, hi] distribution."""
    return [round(lo * (hi / lo) ** ((i + 0.5) / n)) for i in range(n)]


def cdiv(a, b):
    return -(-a // b)


def width_for(max_seq_len):
    n = cdiv(cdiv(max(max_seq_len, 1), BLK), 64)
    return (1 << (n - 1).bit_length()) * 64


def load_side(name, path):
    """The package at ``path`` as module ``name`` (``None``: this checkout's)."""
    if path is None:
        return importlib.import_module("amd_kernels.indexer.indexer_nvfp4_d128_p128_top16")
    path = os.path.abspath(path)
    parent, pkg = os.path.split(path)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class Side:
    def __init__(self, name, pkg, rank, group, dev):
        self.name = name
        self.host = importlib.import_module(pkg.__name__ + ".split4.host")
        self.topk = importlib.import_module(pkg.__name__ + ".topk")
        init = self.host.init_rank_owned
        # a frozen copy whose init_rank_owned takes max_prefill_tokens also builds prefill buffers
        old = {"max_prefill_tokens": 4096} if "max_prefill_tokens" in inspect.signature(init).parameters else {}
        init(rank, WORLD, group, dev, width_for(MAX_MODEL_LEN), MAX_REQS, QL, sets=SETS, **old)
        self.dec = self.host.get_rank_owned(0)
        self.bar = self.host.get_rank_owned(1)
        self.grid = 0  # the scorer's grid (0: the host's choice)
        self.tk = None  # decode_score.TK_MAX_ROWS at the call (None: the package's)

    def arm(self, name, grid=0, tk=None):
        """Another side of the same package and buffers, other dispatch overrides."""
        a = Side.__new__(Side)
        a.__dict__.update(self.__dict__)
        a.name, a.grid, a.tk = name, grid, tk
        return a

    def decode_call(self, inp, out):
        if self.tk is None:
            return self._decode_call(inp, out)
        ds = importlib.import_module(self.host.__name__.rsplit(".", 1)[0] + ".decode_score")
        old, ds.TK_MAX_ROWS = ds.TK_MAX_ROWS, self.tk
        try:
            return self._decode_call(inp, out)
        finally:
            ds.TK_MAX_ROWS = old

    def _decode_call(self, inp, out):
        width = width_for(MAX_MODEL_LEN)
        if hasattr(self.dec, "score_topk"):
            # the scorer with the top-k fused (one launch); older packages: the
            # two calls
            self.dec.score_topk(
                inp["iq"], inp["kv"], inp["table"], inp["seq"], INIT, LOCAL, width, out,
                MAX_MODEL_LEN, grid=self.grid,
            )  # fmt: skip
            return
        score = self.dec.score_push(
            inp["iq"], inp["kv"], inp["table"], inp["seq"], INIT, LOCAL, width, grid=self.grid
        )
        self.topk.index_topk_decode_indices(
            score, out, inp["seq"], max_seq_len=MAX_MODEL_LEN, block_size=BLK,
            query_len=QL, rank_owned_order=True,
        )  # fmt: skip


class Pool:
    """This rank's slots of the global page pool (page 4 s + rank at slot s)."""

    def __init__(self, rank, slots, nvfp4_index_pages):
        self.slots = slots
        self.cache = torch.empty(slots, BLK, 72, dtype=torch.uint8, device="cuda")
        step = 512
        g = torch.Generator(device="cuda")
        for c0 in range(0, slots, step):
            n = min(step, slots - c0)
            g.manual_seed(1_000_003 * rank + c0 // step)
            keys = torch.randn(n * BLK, D, device="cuda", generator=g) * 2
            self.cache[c0 : c0 + n] = nvfp4_index_pages(keys.bfloat16())


def class_counts(lens):
    nb = [cdiv(s, BLK) for s in lens]
    return [sum(max(0, cdiv(w - c, 4)) for w in nb) for c in range(WORLD)]


def make_table(lens, region, pool_slots, rng, cols):
    """Block table [reqs, cols]: block b of a request on page 4 s + b % 4, s
    from class (b % 4)'s run of slots of this input's region, shuffled."""
    need = class_counts(lens)
    span = max(max(need), 1)
    base = region * span
    perms = [((base + rng.permutation(n)) % pool_slots) * 4 + c for c, n in enumerate(need)]
    pos = [0] * WORLD
    t = np.zeros((len(lens), cols), dtype=np.int32)
    for r, s in enumerate(lens):
        w = cdiv(s, BLK)
        for c in range(WORLD):
            k = len(range(c, w, 4))
            t[r, c:w:4] = perms[c][pos[c] : pos[c] + k]
            pos[c] += k
    return torch.from_numpy(t).cuda()


def num_calls(bytes_per_rank, base, regions):
    calls = max(base, 1 + cdiv(REUSE_BYTES, max(1, bytes_per_rank)))
    # Close the rotation at a region boundary. A partial final cycle could
    # replay the same page region immediately across the graph boundary.
    if calls >= regions:
        calls = cdiv(calls, regions) * regions
    assert (min(calls, regions) - 1) * bytes_per_rank >= REUSE_BYTES
    return calls


def decode_inputs(m, ctx, rank, pool, lens=None):
    nreq = m // QL
    lens0 = list(lens) if lens else [ctx] * nreq
    assert len(lens0) == nreq
    per_rank = max(class_counts(lens0)) * PAGE_BYTES
    span = max(max(class_counts(lens0)), 1)
    regions = max(1, pool.slots // span)
    assert regions >= 3, ("pool too small", pool.slots, span)
    calls = num_calls(per_rank, 100, regions)
    out = []
    for i in range(calls):
        rng = np.random.default_rng(1000 * m + i)
        lens = lens0
        q = rng.standard_normal((m, WORLD, D), dtype=np.float32) * 0.5
        iq = torch.from_numpy(np.ascontiguousarray(q[:, rank])).cuda().to(torch.float8_e4m3fn)
        table = make_table(lens, i % regions, pool.slots, rng, MAX_MODEL_LEN // BLK)
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        out.append(dict(iq=iq, kv=pool.cache, table=table, seq=seq))
    return out, per_rank


class Barrier:
    """A host barrier, then one small rank-owned decode call: its end on a rank
    means every rank reached it (the scorer's last workgroup waits for all)."""

    def __init__(self, side, pool, rank):
        rng = np.random.default_rng(7)
        q = rng.standard_normal((QL, WORLD, D), dtype=np.float32) * 0.5
        self.side = side
        self.iq = torch.from_numpy(np.ascontiguousarray(q[:, rank])).cuda().to(torch.float8_e4m3fn)
        self.table = make_table([4 * BLK], 0, pool.slots, rng, 64)
        self.seq = torch.tensor([4 * BLK], dtype=torch.int32, device="cuda")
        self.kv = pool.cache

    def __call__(self):
        dist.barrier()
        self.side.bar.score_push(self.iq, self.kv, self.table, self.seq, INIT, LOCAL, 64)


def gather_max(x):
    t = torch.tensor([x], dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def gather_all(obj):
    out = [None] * WORLD
    dist.all_gather_object(out, obj)
    return out


def kernel_times(run, sync_bar):
    from torch.profiler import ProfilerActivity, profile

    sync_bar()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        # Profiler initialization can skew the four hosts by milliseconds.
        dist.barrier()
        run()
        torch.cuda.synchronize()
    res = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.device_time > 0:
            name = ev.name
            d = res.setdefault(name, [0.0, 0])
            d[0] += ev.device_time
            d[1] += 1
    return res


def bench_cell(kind, sides, inputs, outs, bar, args, rank):
    n = len(inputs)
    runs = {}
    for s in sides:
        call = s.decode_call
        out = outs[s.name]
        for inp in inputs[:2]:
            call(inp, out)
        torch.cuda.synchronize()
        bar()
        # Large cold rotations can have >20K calls. Submitting one
        # >60K-node HIP graph can block graph launch across CP ranks.
        # Use graphs of 100..199 calls (one graph for smaller rotations),
        # keeping the full input order and reuse period unchanged.
        chunks = max(1, n // 100)
        graphs = []
        for chunk in range(chunks):
            lo, hi = chunk * n // chunks, (chunk + 1) * n // chunks
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for inp in inputs[lo:hi]:
                    call(inp, out)
            graphs.append(g)
        torch.cuda.synchronize()
        runs[s.name] = [graph.replay for graph in graphs]
    # checks: every side against the first on the first and last input (rows
    # that differ, the max over ranks; with more than two sides per side)
    check = {}
    if len(sides) >= 2:
        for i in (0, n - 1):
            got = []
            for s in sides:
                call = s.decode_call
                o = outs[s.name]
                o.fill_(-7)
                bar()
                call(inputs[i], o)
                torch.cuda.synchronize()
                got.append(o.clone())
            diffs = [gather_max((got[0] != g).any(-1).sum().item()) for g in got[1:]]
            check[i] = diffs[0] if len(sides) == 2 else diffs
    # warm-up
    for s in sides:
        if args.progress and rank == 0:
            print(f"# warm-up {kind} side={s.name} calls={n}", flush=True)
        bar()
        for _ in range(2):
            for run in runs[s.name]:
                run()
                torch.cuda.synchronize()
    samples = {s.name: [] for s in sides}
    ranks = {s.name: [] for s in sides}  # per round: every rank's time
    for k in range(args.rounds):
        order = sides if k % 2 == 0 else sides[::-1]
        for s in order:
            if args.progress and rank == 0:
                print(f"# timing {kind} M={outs[s.name].shape[-2]} round={k} side={s.name}", flush=True)
            elapsed = 0.0
            for _ in range(args.replays):
                for run in runs[s.name]:
                    # One 100..199-call graph/batch per event pair. A bounded
                    # CP submission avoids back-pressure across HIP queues.
                    # Host/device alignment and the completion wait are outside
                    # each interval; the unit's own exchanges stay inside.
                    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                    bar()
                    st.record()
                    run()
                    en.record()
                    torch.cuda.synchronize()
                    elapsed += st.elapsed_time(en)
            us = elapsed * 1e3 / (args.replays * n)
            ranks[s.name].append(gather_all(us))
            samples[s.name].append(max(ranks[s.name][-1]))
            if args.progress and rank == 0:
                print(f"# completed {samples[s.name][-1]:.3f} us", flush=True)
    prof = {}
    if args.profile:
        probes = inputs[: min(n, 16)]
        for s in sides:
            call = s.decode_call
            out = outs[s.name]

            def profile_run(call=call, out=out):
                for inp in probes:
                    call(inp, out)

            bar()
            small_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(small_graph):
                profile_run()
            torch.cuda.synchronize()
            kt = kernel_times(small_graph.replay, bar)
            tot = sum(v[0] for v in kt.values())
            allk = gather_all((tot, kt))
            slow = max(range(WORLD), key=lambda r: allk[r][0])
            prof[s.name] = {
                "rank": slow,
                "calls": len(probes),
                "kernels": {k: [v[0] / len(probes), v[1] / len(probes)] for k, v in allk[slow][1].items()},
            }
    return samples, check, prof, ranks


def fmt(v):
    return f"{statistics.median(v):.2f} [{min(v):.2f}-{max(v):.2f}]"


def worker(rank, port, args, errfile):
    try:
        with torch.inference_mode():
            body(rank, port, args)
    except BaseException:
        with open(f"{errfile}.{rank}", "w") as f:
            f.write(traceback.format_exc())
        raise


def parse_arms(spec):
    """``name:tk=R,grid=G;...`` -> [(name, grid, tk)]"""
    out = []
    for item in filter(None, (x.strip() for x in (spec or "").split(";"))):
        name, _, opts = item.partition(":")
        kv = dict(o.split("=") for o in filter(None, opts.split(",")))
        tk = kv.get("tk")
        out.append((name, int(kv.get("grid", 0)), None if tk is None else int(tk)))
    return out


def body(rank, port, args):
    global QL
    QL = args.ql
    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    group = dist.group.WORLD
    dev = torch.device("cuda", rank)
    ours = load_side("ours", None)
    # The sides' buffers are built in the timed order (moving a built Side to the front reorders only the timing).
    # --ref PATH (side "today") or NAME=PATH,NAME=PATH,... (several frozen copies)
    refs = [x.split("=", 1) if "=" in x else ("today", x) for x in args.ref.split(",")] if args.ref else []
    specs = refs + [("ours", None)]
    if args.ours_first:
        specs.reverse()
    sides = [Side(name, ours if name == "ours" else load_side("idx_" + name, path), rank, group, dev)
             for name, path in specs]
    sides[-1 if not args.ours_first else 0].grid = args.ours_grid
    base = sides[-1 if not args.ours_first else 0]
    sides += [base.arm(name, grid, tk) for name, grid, tk in parse_arms(args.arms)]
    nvfp4_index_pages = importlib.import_module(ours.__name__ + ".reference").nvfp4_index_pages
    dec_m = [int(x) for x in args.decode.split(",")] if args.decode else []
    ctxs = [int(x) for x in args.ctx.split(",")] if args.ctx else []
    lens_by_n = {}
    if args.lens_json:
        table = json.load(open(args.lens_json))
        lens_by_n = {int(n): table[n] for n in args.reqs.split(",")}
    elif args.reqs:
        lens_by_n = {int(n): synthetic_lens(int(n)) for n in args.reqs.split(",")}
        if rank == 0:
            print(f"# synthetic lengths (no --lens-json): log-uniform {SYNTHETIC_RANGE[0]}-{SYNTHETIC_RANGE[1]} "
                  "quantiles", flush=True)
    biggest = max(
        [max(class_counts([c] * (m // QL))) for m in dec_m for c in ctxs]
        + [max(class_counts(v)) for v in lens_by_n.values()]
        + [1]
    )
    slots = max(cdiv(MIN_POOL_BYTES, PAGE_BYTES), 3 * biggest + 8)
    t0 = time.time()
    pool = Pool(rank, slots, nvfp4_index_pages)
    torch.cuda.synchronize()
    bar = Barrier(sides[0], pool, rank)
    if rank == 0:
        print(f"# pool {slots} slots ({slots * PAGE_BYTES / 2**20:.0f} MB per rank) in {time.time() - t0:.1f} s",
              flush=True)  # fmt: skip
    results = []
    cells = [("decode", m, c) for c in ctxs for m in dec_m] + [
        ("decode", n * QL, f"lens{n}") for n in lens_by_n]
    for kind, m, c in cells:
        t0 = time.time()
        lens = lens_by_n.get(m // QL) if isinstance(c, str) else None
        inputs, per_rank = decode_inputs(m, c, rank, pool, lens)
        outs = {s.name: torch.empty(1, m, TOPK, dtype=torch.int32, device="cuda") for s in sides}
        for mode in ["graph"]:
            samples, check, prof, ranks = bench_cell(kind, sides, inputs, outs, bar, args, rank)
            rec = dict(
                kind=kind, mode=mode, batches_per_period=max(1, len(inputs) // 100), M=m, ctx=c, calls=len(inputs), page_mb_per_rank=per_rank / 2**20,
                samples=samples, check=check, prof=prof, secs=time.time() - t0, ql=QL, ranks=ranks,
            )  # fmt: skip
            results.append(rec)
            if rank == 0 and args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            if rank == 0:
                line = f"{kind:7s} {mode:5s} ql={QL} M={m:6d} C={c:>7} calls={len(inputs):4d} "
                line += " ".join(f"{s}={fmt(v)}" for s, v in samples.items())
                if len(samples) == 2 and "today" in samples and "ours" in samples:
                    a, b = samples["today"], samples["ours"]
                    line += f" ours/today={statistics.median(b) / statistics.median(a):.4f}"
                    line += f" diffrows={check}"
                elif len(samples) > 2:
                    first = statistics.median(samples[sides[0].name])
                    line += " /" + sides[0].name + " " + " ".join(
                        f"{k}={statistics.median(v) / first:.4f}" for k, v in list(samples.items())[1:])
                    line += f" diffrows={check}"
                print(line, flush=True)
                for k, v in ranks.items():
                    per = [statistics.median(r[j] for r in v) for j in range(WORLD)]
                    print(f"    {k} per-rank median " + " / ".join(f"{x:.2f}" for x in per), flush=True)
                for s, p in prof.items():
                    for k, v in sorted(p["kernels"].items(), key=lambda kv: -kv[1][0]):
                        print(f"    {s} r{p['rank']} {v[0]:9.2f} us x{v[1]:.0f} {k}", flush=True)
        del inputs
        torch.cuda.empty_cache()
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decode", default=",".join(map(str, DECODE_M)), help="decode M (rows), '' = none")
    ap.add_argument("--ctx", default=",".join(map(str, DECODE_CTX)))
    ap.add_argument("--ref", default=None, help="a frozen copy of the package, timed as 'today' (or NAME=PATH,...)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--replays", type=int, default=10)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--progress", action="store_true", help="log timed block boundaries outside the timed interval")
    ap.add_argument("--ours-grid", type=int, default=0, help="ours' decode scorer grid (0: the host's)")
    ap.add_argument("--ours-first", action="store_true", help="make ours' buffers first (A/A order check)")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    ap.add_argument("--ql", type=int, default=8, help="decode rows per request (1 + the draft length)")
    ap.add_argument("--lens-json", default=None, help="JSON {N: [context per request]}, with --reqs")
    ap.add_argument("--reqs", default="", help="request counts N to time (--lens-json, else synthetic_lens)")
    ap.add_argument("--arms", default="", help="extra sides of this checkout: name:tk=R,grid=G;...")
    args = ap.parse_args()
    assert torch.cuda.device_count() >= WORLD, "needs four GPUs"
    import torch.multiprocessing as mp

    errfile = f"/tmp/bench_rank_owned_err_{os.getpid()}"
    try:
        mp.spawn(worker, args=(free_port(), args, errfile), nprocs=WORLD, join=True)
    except Exception:
        for r in range(WORLD):
            if os.path.exists(f"{errfile}.{r}"):
                print(open(f"{errfile}.{r}").read(), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
