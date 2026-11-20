# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-head decode dispatch's crossovers (``score_decode.host``): every scorer pinned (``PIN_SCORER``; the run
scorer at ``runs:GRID:UNITS``), then both selectors (``PIN_PLAIN``) on the scorer's rows, timed as
``benchmark_index_decode_h2.py`` times (graph replay, a page pool above the MALL, max_seq_len 1M). The scores of the
first input are checked against ``reference.decode_scores_reference``, the ids against ``topk_reference``.

    python3 benchmarks/indexer/indexer_nvfp4_d128_p128_top16/sweep_decode_h2.py --ql 8 --reqs 8,12,13,16,32 \
        --scorers first_page_xcd,runs:512:256,chunked
"""

import argparse
import statistics

import torch

from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    SCORE_ATOL,
    SCORE_RTOL,
    cdiv,
    decode_scores_reference,
    score_width,
    topk_reference,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host
from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_decode_topk import synthetic_lens
from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_index_decode_h2 import capture, time_graph
from benchmarks.indexer.indexer_nvfp4_d128_p128_top16.benchmark_indexer import (
    PAGE_BYTES,
    num_copies,
    page_pool,
    pool_pages_for,
    tables,
)

MSL = 1 << 20


def pin(scorer):
    kind, *grid = scorer.split(":")
    host.PIN_SCORER = kind
    host.PIN_RUNS_GRID, host.PIN_RUNS_UNITS = (int(x) for x in grid) if grid else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--ql", type=int, default=8)
    ap.add_argument("--reqs", default="1,2,4,8,12,13,16,24,32,48,64")
    ap.add_argument("--kinds", default="mix,32K,128K", help="mix: log-uniform 32K-192K quantiles; NK: N K tokens")
    ap.add_argument("--scorers", default="page,first_page_xcd,runs:512:256,chunked")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()
    from amd_kernels import warm_up

    warm_up.set_auto(False)
    heads, ql, scorers = args.heads, args.ql, args.scorers.split(",")
    print("requests | contexts | " + " | ".join(scorers) + " | plain | one-wave | fastest scorer", flush=True)
    for n in (int(x) for x in args.reqs.split(",")):
        for kind in args.kinds.split(","):
            lens = synthetic_lens(n) if kind == "mix" else [int(kind[:-1]) * 1024] * n
            rows = n * ql
            copies = num_copies(sum(cdiv(s, BLK) for s in lens) * PAGE_BYTES)
            pool = pool_pages_for(lens)
            cache = page_pool(pool)
            tabs = tables(lens, copies, pool)
            qs = [(torch.randn(rows, heads, 128, device="cuda") * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
            seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
            kv = (seq.long().repeat_interleave(ql) - ql + torch.arange(rows, device="cuda") % ql + 1).clamp(min=0)
            nblk = (kv + BLK - 1) // BLK
            score = torch.empty(heads, rows, score_width(MSL), device="cuda")
            us = {}
            for sc in scorers:
                pin(sc)
                try:
                    host._score_rows(qs[0], cache, score, tabs[0], seq, 0, 1, ql, MSL)
                except AssertionError:  # a scorer past its page-table range
                    us[sc] = float("nan")
                    continue
                want = decode_scores_reference(qs[0], cache, tabs[0], lens, 0, 1, ql)
                for r, s in enumerate(lens):
                    got = score[:, r * ql:(r + 1) * ql, :cdiv(s, BLK)].double()
                    torch.testing.assert_close(got, want[r], rtol=SCORE_RTOL, atol=SCORE_ATOL)
                g = capture([lambda i=i: host._score_rows(qs[i], cache, score, tabs[i], seq, 0, 1, ql, MSL)
                             for i in range(copies)])
                us[sc] = statistics.median(time_graph(g, copies, 10) for _ in range(args.rounds))
                del g
            pin("chunked")
            host.PIN_SCORER = None
            bufs = [torch.empty_like(score) for _ in range(16)]
            for j, b in enumerate(bufs):
                host._score_rows(qs[j], cache, b, tabs[j], seq, 0, 1, ql, MSL)
            ids = torch.empty(heads, rows, 16, dtype=torch.int32, device="cuda")
            for plain in (True, False):
                host.PIN_PLAIN = plain
                host.nvfp4_index_top16_from_scores(bufs[0], seq, ids, rows=rows, query_len=ql, max_seq_len=MSL)
                assert torch.equal(ids, topk_reference(bufs[0], nblk, MSL))
                g = capture([lambda b=b: host.nvfp4_index_top16_from_scores(b, seq, ids, rows=rows, query_len=ql,
                                                                            max_seq_len=MSL) for b in bufs] * 4)
                us[plain] = statistics.median(time_graph(g, 64, 10) for _ in range(args.rounds))
                del g
            host.PIN_PLAIN = None
            best = min(scorers, key=lambda s: us[s] if us[s] == us[s] else float("inf"))
            print(f"{n} | {kind} | " + " | ".join(f"{us[s]:.2f}" for s in scorers)
                  + f" | {us[True]:.2f} | {us[False]:.2f} | {best}", flush=True)
            del cache, tabs, qs, bufs
            torch.cuda.empty_cache()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
