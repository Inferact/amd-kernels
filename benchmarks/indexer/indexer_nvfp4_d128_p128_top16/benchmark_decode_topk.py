# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-GPU graph-replay timing of the decode top-16 selectors on crafted score
rows, frozen copies of the package vs this checkout (``--selector``):

- ``rank_owned``: ``topk.index_topk_decode_indices(..., rank_owned_order=True)``
  (the split4 chain's second launch; the body of the fused scorer's tail);
- ``plain``: ``score_decode.nvfp4_index_top16_from_scores`` (the one-GPU
  route's plain-order four-wave selector, <= 128 rows, one head);
- ``attn``: the served consumer, ``nvfp4_sparse_decode(..., index_score=)``
  (the attention's ``_select16`` kernels run the plain selector in their
  prologue); ``--attn-ref NAME=PATH``: a frozen copy of the attention package
  whose ``amd_kernels.indexer.indexer_nvfp4_d128_p128_top16`` imports were
  renamed to ``idx_NAME`` (the frozen indexer of ``--ref NAME=...``).

Rows: ``--ql`` rows per request, N requests of the contexts ``--lens[str(N)]``:
``--lens F``, a JSON file {"N": [context length of request i of N, ...]} (e.g.
the (i + 0.5) / N quantiles of a workload's request context lengths); without
it, ``synthetic_lens``: request i of N at the (i + 0.5) / N quantile of a
log-uniform 32K-192K distribution. ``--long L``: the last request gets L tokens.
Scores per kind, in block order (placed in the rank-owned order for
``rank_owned``): ``gauss``; ``real`` (outliers, init 1e30, last two 1e29);
``step`` (the last 15% of the row 3 sigma up), ``ramp`` (a 6-sigma linear
trend), ``rising`` (score = position): served scores rise toward the row's end.
100 score sets per graph (> 256 MB MALL for the larger cells), alternated rounds;
ids checked against ``reference.topk_reference`` on the first set and between
the sides on the last.

    python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_decode_topk.py \
        --ref today=/path/to/frozen/indexer_nvfp4_d128_p128_top16 \
        --lens lengths.json \
        --reqs 16,32,48 --ql 4 --kinds gauss,real,step,ramp,rising
"""

import argparse
import importlib
import importlib.util
import json
import os
import statistics
import sys

import torch

BLK = 128
MSL = 1 << 20
WIDTH = 8192
SYNTHETIC_RANGE = (32768, 196608)  # the log-uniform context range without --lens


def synthetic_lens(n, lo=SYNTHETIC_RANGE[0], hi=SYNTHETIC_RANGE[1]):
    """Request i of n at the (i + 0.5) / n quantile of a log-uniform [lo, hi] distribution."""
    return [round(lo * (hi / lo) ** ((i + 0.5) / n)) for i in range(n)]


def load_side(name, path):
    if path is None:
        return importlib.import_module("amd_kernels.indexer.indexer_nvfp4_d128_p128_top16")
    path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(path, "__init__.py"), submodule_search_locations=[path]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rows_by_block(kind, nblk, gen):
    """[rows, WIDTH] scores in block order; past a row's count: garbage."""
    rows = nblk.numel()
    b = torch.arange(WIDTH, device="cuda")[None, :].float()
    n = nblk[:, None].float().clamp(min=1)
    s = torch.randn(rows, WIDTH, device="cuda", generator=gen)
    if kind == "real":  # outliers, init block 1e30, last two 1e29
        s = torch.where(torch.rand(s.shape, device="cuda", generator=gen) < 4.0 / WIDTH, s + 6, s)
    elif kind == "step":  # the last 15% of the row 3 sigma up
        s = s + 3.0 * (b >= 0.85 * n).float()
    elif kind == "ramp":  # a linear trend of 6 sigma over the row
        s = s + 6.0 * b / n
    elif kind == "rising":
        s = b.expand(rows, WIDTH).clone()
    if kind in ("real", "step", "ramp"):
        idx = torch.arange(rows, device="cuda")
        last = (nblk - 1).clamp(min=0)
        s[idx, last] = 1e29
        s[idx, (nblk - 2).clamp(min=0)] = 1e29
        s[:, 0] = 1e30
    live = b < nblk[:, None]
    return torch.where(live, s, torch.full_like(s, 1e6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="name=path,... (the first: the base) or one path (today)")
    ap.add_argument("--attn-ref", default="", help="--selector attn: name=path,... of frozen attention copies")
    ap.add_argument("--selector", default="rank_owned", choices=("rank_owned", "plain", "attn"))
    ap.add_argument("--lens", default=None, help='JSON {"N": [context per request]}; default: synthetic_lens')
    ap.add_argument("--reqs", default="32")
    ap.add_argument("--long", type=int, default=0, help="the last request's context (0: the table's)")
    ap.add_argument("--ql", type=int, default=4)
    ap.add_argument("--kinds", default="gauss,real,step,ramp,rising")
    ap.add_argument("--inputs", type=int, default=100)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--replays", type=int, default=10)
    args = ap.parse_args()
    from amd_kernels import warm_up

    warm_up.set_auto(False)  # every variant compiles at its first (untimed) call
    if args.lens:
        table = json.load(open(args.lens))
    else:
        table = {str(n): synthetic_lens(n) for n in (int(x) for x in args.reqs.split(","))}
        print(f"synthetic lengths (no --lens): log-uniform {SYNTHETIC_RANGE[0]}-{SYNTHETIC_RANGE[1]} quantiles",
              flush=True)
    ours = load_side("ours", None)
    refs = [x.split("=", 1) if "=" in x else ("today", x) for x in args.ref.split(",")]
    pkgs = {name: load_side("idx_" + name, path) for name, path in refs}
    pkgs["ours"] = ours
    attn = {}
    if args.selector == "attn":
        for name, path in (x.split("=", 1) for x in filter(None, args.attn_ref.split(","))):
            attn[name] = importlib.import_module(load_side("attn_" + name, path).__name__ + ".host")
        attn["ours"] = importlib.import_module("amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.host")
        pkgs = {k: v for k, v in pkgs.items() if k in attn}
    base = refs[0][0]
    ref = importlib.import_module(ours.__name__ + ".reference")
    pos = ref.rank_owned_position(torch.arange(WIDTH, device="cuda"))
    ql = args.ql
    for n in [int(x) for x in args.reqs.split(",")]:
        lens_l = list(table[str(n)])
        if args.long:
            lens_l[-1] = args.long
        lens = torch.tensor(lens_l, dtype=torch.int32, device="cuda")
        rows = n * ql
        kv = (lens.repeat_interleave(ql) - ql + torch.arange(rows, device="cuda") % ql + 1).clamp(min=0)
        nblk = (kv + BLK - 1) // BLK
        if args.selector == "attn":
            g0 = torch.Generator(device="cuda").manual_seed(5)
            pages = 4096
            kv_cache = torch.randint(0, 256, (pages, 2, BLK, 72), dtype=torch.uint8, device="cuda",
                                     generator=g0) & 0x77
            bt = torch.randint(0, pages, (n, MSL // BLK), dtype=torch.int32, device="cuda", generator=g0)
            q = torch.randn(rows, 16, 128, device="cuda", generator=g0).bfloat16()
            one = torch.ones((), dtype=torch.float32, device="cuda")
            pad = (rows + 31) // 32 * 32
            bufs = {k: (torch.zeros(rows, 16, 128, dtype=torch.bfloat16, device="cuda"),
                        torch.zeros(pad, 16 * 128, dtype=torch.uint8, device="cuda"),
                        torch.zeros(pad, 16 * 128 // 32, dtype=torch.uint8, device="cuda"))
                    for k in pkgs}

        def make_call(k):
            if args.selector == "rank_owned":
                fn = importlib.import_module(pkgs[k].__name__ + ".topk").index_topk_decode_indices
                return lambda s, o: fn(s, o, lens, max_seq_len=MSL, query_len=ql, rank_owned_order=True)
            if args.selector == "plain":
                fn = importlib.import_module(pkgs[k].__name__ + ".score_decode").nvfp4_index_top16_from_scores
                return lambda s, o: fn(s, lens, o, rows=rows, query_len=ql, max_seq_len=MSL)
            h, (out, codes, scales) = attn[k], bufs[k]
            return lambda s, o: h.nvfp4_sparse_decode(
                q, kv_cache, o, bt, lens, 128 ** -0.5, out, k_scale=one, v_scale=one,
                decode_query_len=ql, quant_out=(codes, scales, 0), index_score=s,
                index_max_seq_len=MSL)

        for kind in args.kinds.split(","):
            gen = torch.Generator(device="cuda").manual_seed(17 + n)
            scores = []
            first_block = None
            for i in range(args.inputs):
                by_block = rows_by_block(kind, nblk, gen)
                if i == 0:
                    first_block = by_block.clone()
                if args.selector == "rank_owned":
                    s = torch.empty(1, rows, WIDTH, device="cuda")
                    s[0][:, pos] = by_block
                else:
                    s = by_block[None].contiguous()
                scores.append(s)
            outs = {k: torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda") for k in pkgs}
            calls = {k: make_call(k) for k in pkgs}
            graphs = {}
            for k, call in calls.items():
                call(scores[0], outs[k])
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for s in scores:
                        call(s, outs[k])
                graphs[k] = g
            # check: every side on the first input against the reference
            want = ref.topk_reference(first_block[None], nblk, MSL)
            ok = {}
            for k, call in calls.items():
                outs[k].fill_(-7)
                call(scores[0], outs[k])
                torch.cuda.synchronize()
                ok[k] = bool(torch.equal(outs[k], want))
            for k in graphs:
                graphs[k].replay()
            torch.cuda.synchronize()
            ok_last = all(bool(torch.equal(outs[base], o)) for o in outs.values())
            if args.selector == "attn":
                first = next(iter(bufs.values()))
                # the MXFP8 output rows (codes, scales) of the last set, bit for bit
                ok_last = ok_last and all(torch.equal(first[1][:rows], b[1][:rows])
                                          and torch.equal(first[2][:rows], b[2][:rows])
                                          for b in bufs.values())
            samples = {k: [] for k in pkgs}
            for r in range(args.rounds):
                names = list(pkgs)
                order = names[r % len(names):] + names[:r % len(names)]
                if r % 2:
                    order = order[::-1]
                for k in order:
                    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
                    st.record()
                    for _ in range(args.replays):
                        graphs[k].replay()
                    en.record()
                    torch.cuda.synchronize()
                    samples[k].append(st.elapsed_time(en) * 1e3 / (args.replays * args.inputs))
            med = {k: statistics.median(v) for k, v in samples.items()}
            print(
                f"{args.selector} ql={ql} N={n} rows={rows} max_blocks={int(nblk.max())} kind={kind:6s} "
                + " ".join(f"{k}={med[k]:.2f} [{min(v):.2f}-{max(v):.2f}]" for k, v in samples.items())
                + " " + " ".join(f"{k}/{base}={med[k] / med[base]:.4f}" for k in med if k != base)
                + f" ref_ok={all(ok.values())} last_equal={ok_last}",
                flush=True,
            )
            del scores, graphs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
