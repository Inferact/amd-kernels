# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-GPU graph-replay timing of the decode index top-16 at two index heads (TP2: 2 index-q heads per GPU over the
one replicated index-K head), this checkout against frozen copies of the package (``--ref NAME=PATH``).

Routes as the TP2 decode block runs them (``attention/.../index_decode.py``): ``chain`` = ``nvfp4_index_score_top16``
(more than 256 rows), ``fused`` = ``nvfp4_index_score_rows`` then ``nvfp4_index_top16_from_scores`` (what the attention
runs at two KV heads, ``host._select_ids``). Every call has its own index query and block table into its own pages of
a pool above the 256 MB MALL; contexts per request count: ``--lens`` JSON {"N": [...]} or the (i + 0.5) / N quantiles
of a log-uniform 32K-192K distribution, plus uniform 32K / 128K. max_seq_len = 1M (what decode graphs capture). The
ids of the first and last input are checked against ``reference.topk_reference`` and between the sides.

    python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/benchmark_index_decode_h2.py \
        --ref base=/path/to/frozen/indexer_nvfp4_d128_p128_top16 --ql 8 --reqs 1,2,4,8,16,32,64
"""

import argparse
import importlib
import json
import statistics

import torch

from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_decode_topk import load_side, synthetic_lens
from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_indexer import (
    PAGE_BYTES,
    num_copies,
    page_pool,
    pool_pages_for,
    tables,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import BLK, cdiv, score_width, topk_reference

MSL = 1 << 20
INIT, LOCAL = 0, 1


def capture(calls):
    """One graph of ``calls`` (each run once eagerly first), warmed by three replays."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    torch.cuda.synchronize()
    for _ in range(3):
        g.replay()
    return g


def time_graph(g, n_calls, replays):
    """Mean time per call (us) over ``replays`` replays of ``g``."""
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(replays):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / (replays * n_calls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="", help="name=path,... frozen copies of the package")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ql", type=int, default=8)
    ap.add_argument("--reqs", default="1,2,3,4,6,8,12,16,24,32,48,64")
    ap.add_argument("--lens", default=None, help='JSON {"N": [context per request]}')
    ap.add_argument("--kinds", default="mix,32K,128K")
    ap.add_argument("--routes", default="fused,chain")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--replays", type=int, default=10)
    args = ap.parse_args()
    from amd_kernels import warm_up

    warm_up.set_auto(False)
    sides = {"ours": load_side("ours", None)}
    importlib.import_module(sides["ours"].__name__ + ".score_decode")
    for x in filter(None, args.ref.split(",")):
        name, path = x.split("=", 1)
        registry = dict(warm_up.REGISTRY)
        warm_up.REGISTRY.clear()  # a frozen copy registers the same entry names
        sides[name] = load_side("idx_" + name, path)
        importlib.import_module(sides[name].__name__ + ".score_decode")
        warm_up.REGISTRY.clear()
        warm_up.REGISTRY.update(registry)
    mods = {k: importlib.import_module(v.__name__ + ".score_decode") for k, v in sides.items()}
    table = json.load(open(args.lens)) if args.lens else {}
    heads, ql = args.heads, args.ql
    print(f"| route | heads x rows | requests | contexts | " + " | ".join(f"{k} us" for k in sides)
          + " | " + " | ".join(f"{k}/ours" for k in sides if k != "ours") + " | page GB/s (ours) |", flush=True)
    for n in [int(x) for x in args.reqs.split(",")]:
        for kind in args.kinds.split(","):
            if kind == "mix":
                lens = list(table[str(n)]) if str(n) in table else synthetic_lens(n)
            else:
                lens = [int(kind[:-1]) * 1024] * n
            rows = n * ql
            page_bytes = sum(cdiv(s, BLK) for s in lens) * PAGE_BYTES
            copies = num_copies(page_bytes)
            pool = pool_pages_for(lens)
            cache = page_pool(pool)
            tabs = tables(lens, copies, pool)
            gen = torch.Generator(device="cuda").manual_seed(n)
            qs = [(torch.randn(rows, heads, 128, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn)
                  for _ in range(copies)]
            seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
            kv = (seq.long().repeat_interleave(ql) - ql + torch.arange(rows, device="cuda") % ql + 1).clamp(min=0)
            nblk = (kv + BLK - 1) // BLK
            for route in args.routes.split(","):
                outs = {k: torch.full((heads, rows, 16), -7, dtype=torch.int32, device="cuda") for k in sides}
                scores = {k: torch.empty(heads, rows, score_width(MSL), device="cuda") for k in sides}

                def call(k, i):
                    m = mods[k]
                    if route == "chain":
                        m.nvfp4_index_score_top16(qs[i], cache, tabs[i], seq, INIT, LOCAL, outs[k], query_len=ql,
                                                  max_seq_len=MSL, score=scores[k])
                    else:
                        m.nvfp4_index_score_rows(qs[i], cache, scores[k], tabs[i], seq, INIT, LOCAL, query_len=ql,
                                                 max_seq_len=MSL)
                        m.nvfp4_index_top16_from_scores(scores[k], seq, outs[k], rows=rows, query_len=ql,
                                                        max_seq_len=MSL)

                for i in (0, copies - 1):
                    for k in sides:
                        call(k, i)
                    torch.cuda.synchronize()
                    want = topk_reference(scores["ours"], nblk, MSL)
                    for k in sides:
                        assert torch.equal(outs[k], want), (k, route, n, kind)
                graphs = {k: capture([lambda k=k, i=i: call(k, i) for i in range(copies)]) for k in sides}
                samples = {k: [] for k in sides}
                names = list(sides)
                for r in range(args.rounds):
                    for k in (names if r % 2 == 0 else names[::-1]):
                        samples[k].append(time_graph(graphs[k], copies, args.replays))
                med = {k: statistics.median(v) for k, v in samples.items()}
                gbs = page_bytes / (med["ours"] * 1e-6) / 1e9
                print(f"| {route} | h{heads} x {ql} | {n} | {kind} | " + " | ".join(f"{med[k]:.2f}" for k in sides)
                      + " | " + " | ".join(f"{med[k] / med['ours']:.3f}" for k in sides if k != "ours")
                      + f" | {gbs:.0f} |", flush=True)
                del graphs
            del cache, tabs, qs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
