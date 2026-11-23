# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the MXFP4 indexer against vLLM's decode path over AITER (``paged_mxfp4_mqa_logits`` with the
step's schedule, ``top_k_per_row_decode``: the HIP op below 128 rows, AITER's from 128, then the ragged slot lists of
``compute_global_topk_ragged_indices_and_indptr``).

``--mode dense``: a dense layer (``indexer_topk`` with the ragged lists, against logits + top-512 + the ragged lists);
``source``: the candidate source layer (also the 2048 candidate blocks: against the walk that writes logits and block
maxima, the block top-2048 and the key top-512, + the ragged lists); ``candidates``: a candidate consumer
(``indexer_candidate_topk`` + ragged lists, against ``build_candidate_gather``, the gathered logits, the top-512,
``_remap_compact_topk``, the ragged lists).

One decode step of ``--reqs`` requests of ``--next-n`` rows. Request i of N has the context ``ctx * (spread + (1 -
spread) * (i + 0.5) / N)`` entries (``--spread`` 0.75: the agentic workload's 100-130k-entry contexts at 130k). 100
distinct inputs per graph (q, scales, weights, page tables, lengths) over one cache pool larger than the LLC; median of
5 rounds x 5 replays, us per call, rounds A B B A. The AITER logits are as wide as vLLM allocates them
(``--logits-width``: max_model_len // compress ratio); its step schedule is built once per step for the dense layers
(four), so a quarter of it is counted per call. KV MB: the bytes the dense walk reads; ideal us at 8 TB/s HBM. ``--ctx 0``: the agentic mix of contexts (``MIX_CDF``), ``--max-ctx``: the static bound.
Run: python3 benchmarks/indexer/indexer_mxfp4_h32_d128_top512/benchmark_indexer.py [--mode dense|source|candidates]
"""

import argparse
import statistics

import torch
from amd_kernels.indexer import indexer_mxfp4_h32_d128_top512 as ix
from amd_kernels.indexer.indexer_mxfp4_h32_d128_top512.common import CANDIDATES, ENTRY_BYTES, H, TOPK

NIN = 100
SLOT_BLOCK = 128


def timed(g):
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * NIN))
    return statistics.median(ts)


def graph(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def abba(graphs: dict, rounds: int = 2) -> dict:
    t = {k: [] for k in graphs}
    order = list(graphs)
    for r in range(rounds):
        for k in order if r % 2 == 0 else order[::-1]:
            t[k].append(timed(graphs[k]))
    return {k: statistics.median(v) for k, v in t.items()}


# the agentic workload's entries per request (ratio 1): quantile -> entries, mean ~130k, p95 ~420k, peak ~1M
MIX_CDF = [(0.0, 4000), (0.25, 40000), (0.5, 90000), (0.75, 150000), (0.9, 300000), (0.95, 420000), (0.99, 800000), (1.0, 1000000)]


def contexts(ctx: int, reqs: int, spread: float) -> list[int]:
    if ctx == 0:  # --ctx 0: the agentic mix
        out = []
        for i in range(reqs):
            u = (i + 0.5) / reqs
            (q0, c0), (q1, c1) = next(((a, b) for a, b in zip(MIX_CDF, MIX_CDF[1:]) if u <= b[0]))
            out.append(int(c0 + (c1 - c0) * (u - q0) / (q1 - q0)))
        return out
    return [int(ctx * (spread + (1 - spread) * (i + 0.5) / reqs)) for i in range(reqs)]


def make_inputs(lengths, next_n, P, pool, dev, gen):
    """One distinct decode step over ``pool`` [pages, P, 68]: random page tables, q, scales, weights; the attention
    cache's page table and the valid-token flags of the ragged lists."""
    B = len(lengths)
    npg = [-(-c // P) for c in lengths]
    ids = torch.randperm(pool.shape[0], device=dev, generator=gen)[: sum(npg)].int()
    bt = torch.zeros(B, max(npg), dtype=torch.int32, device=dev)
    o = 0
    for i, n in enumerate(npg):
        bt[i, :n] = ids[o : o + n]
        o += n
    T = B * next_n
    q = torch.randint(0, 256, (T, H, 64), dtype=torch.uint8, device=dev, generator=gen)
    qs = torch.randint(120, 127, (T, H, 4), dtype=torch.uint8, device=dev, generator=gen)
    w = torch.randn(T, H, device=dev, generator=gen) * 0.1
    cl = torch.tensor(lengths, dtype=torch.int32, device=dev)
    ends = (cl[:, None] - next_n + torch.arange(next_n, device=dev)[None] + 1).clamp(min=0).reshape(-1).int()
    slot_bt = torch.randint(
        0, 1 << 20, (B, max(-(-c // SLOT_BLOCK) for c in lengths)), dtype=torch.int32, device=dev, generator=gen
    )
    return dict(
        q=q,
        qs=qs,
        w=w,
        bt=bt,
        cl=cl,
        ends=ends.contiguous(),
        slot_bt=slot_bt,
        valid=torch.ones(T, dtype=torch.int32, device=dev),
    )


def vllm_ragged(topk, req_of_row, x):
    from vllm.models.deepseek_v41.amd.rocm import _compute_global_topk_ragged_indices_and_indptr_old

    return _compute_global_topk_ragged_indices_and_indptr_old(topk, req_of_row, x["slot_bt"], SLOT_BLOCK, x["valid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "source", "candidates"], default="dense")
    ap.add_argument("--ctx", type=int, nargs="+", default=[131072, 65536])
    ap.add_argument("--reqs", type=int, nargs="+", default=[11, 32, 64, 128])
    ap.add_argument("--next-n", type=int, default=6)
    ap.add_argument("--spread", type=float, default=0.75)
    ap.add_argument("--logits-width", type=int, help="default: 1048576 (ratio 1) or 524288 (ratio 2)")
    ap.add_argument("--max-ctx", type=int, help="the static bound passed to the kernels (default: --ctx, or 1048576 for --ctx 0)")
    ap.add_argument("--pool-gb", type=float, default=1.5)
    ap.add_argument("--no-ragged", action="store_true", help="time the top-512 only")
    args = ap.parse_args()
    from aiter.ops.triton.attention import pa_mqa_logits_mxfp4 as pa
    from vllm.v1.attention.ops import rocm_paged_mxfp4_indexer as vi

    dev = torch.device("cuda", 0)
    gen = torch.Generator(device=dev).manual_seed(1)
    nn = args.next_n
    ragged_on = not args.no_ragged
    print("ctx | reqs | rows | KV MB | ideal us | baseline us | this us | speedup | baseline parts")
    for ctx in args.ctx:
        bound = args.max_ctx or (ctx or 1048576)
        P = 128 if (ctx > 100000 or ctx == 0) else 64
        pages = int(args.pool_gb * 2**30 / (P * ENTRY_BYTES))
        pool = ix.random_cache(pages, P, dev, seed=7) if hasattr(ix, "random_cache") else None
        if pool is None:
            from amd_kernels.indexer.indexer_mxfp4_h32_d128_top512.reference import random_cache

            pool = random_cache(pages, P, dev, seed=7)
        kv4 = pool.view(pool.shape[0], P, 1, ENTRY_BYTES)
        width = args.logits_width or (1048576 if P == 128 else 524288)
        for reqs in args.reqs:
            lengths = contexts(ctx, reqs, args.spread)
            lengths = [min(c, bound) for c in lengths]
            if sum(-(-c // P) for c in lengths) > pages:
                print(f"{ctx} | {reqs} | pool too small")
                continue
            T = reqs * nn
            xs = [make_inputs(lengths, nn, P, pool, dev, gen) for _ in range(NIN)]
            req_of_row = torch.arange(T, device=dev, dtype=torch.int32) // nn
            logits = torch.empty(T, width, dtype=torch.float32, device=dev)
            nblk = (width + 7) // 8
            bscores = torch.empty(T, nblk, dtype=torch.float32, device=dev) if args.mode == "source" else None
            sbuf = torch.empty(4 * pa.SCHED_SLOT_CAP, dtype=torch.int32, device=dev)
            topk = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
            cand = torch.full((T, CANDIDATES), -1, dtype=torch.int32, device=dev)
            out = torch.empty(T, TOPK, dtype=torch.int32, device=dev)
            rg = [
                ix.Ragged(
                    torch.empty(T * TOPK, dtype=torch.int32, device=dev),
                    torch.empty(T + 1, dtype=torch.int32, device=dev),
                    torch.empty(T, dtype=torch.int32, device=dev),
                    x["valid"],
                    x["slot_bt"],
                    SLOT_BLOCK,
                )
                for x in xs
            ]
            q4 = [(x["q"].view(reqs, nn, H, 64), x["qs"].view(reqs, nn, H, 4)) for x in xs]
            sched = [
                pa.build_schedule(x["cl"], nn, H, 128, P, out=sbuf, row_ends=x["ends"], max_model_len=width) for x in xs
            ]
            sched = [None if s is None else s.clone() for s in sched]

            def schedule():
                for x in xs:
                    pa.build_schedule(x["cl"], nn, H, 128, P, out=sbuf, row_ends=x["ends"], max_model_len=width)

            def walk():
                for x, (a, b), s in zip(xs, q4, sched):
                    kw = {}
                    if args.mode == "source":
                        kw = dict(calc_block_scores=True, block_scores=bscores, candidate_block_size=8)
                    pa.paged_mxfp4_mqa_logits(
                        a,
                        b,
                        kv4,
                        x["w"],
                        x["cl"],
                        x["bt"],
                        width,
                        out_logits=logits,
                        clean_logits=False,
                        row_ends=x["ends"],
                        schedule=s,
                        **kw,
                    )

            def selects():
                for x in xs:
                    if args.mode == "source":
                        vi._topk(bscores, (x["ends"] + 14) // 8, cand, CANDIDATES, T)
                    vi._topk(logits, x["ends"], topk, TOPK, T)

            def pack():
                for x in xs:
                    vllm_ragged(topk, req_of_row, x)

            graphs = {"walk": graph(walk), "select": graph(selects), "schedule": graph(schedule)}
            if ragged_on:
                graphs["pack"] = graph(pack)

            if args.mode == "candidates":
                for x in xs:
                    ix.indexer_topk(
                        x["q"],
                        x["qs"],
                        x["w"],
                        pool,
                        x["bt"],
                        x["ends"],
                        out,
                        next_n=nn,
                        max_ctx=bound,
                        candidates_out=cand,
                    )
                    x["cand"] = cand.clone()
                btr = [x["bt"].repeat_interleave(nn, 0).contiguous() for x in xs]
                gath = [pa.build_candidate_gather(x["cand"], x["ends"], b, kv4, H, 128, 8) for x, b in zip(xs, btr)]
                compact = torch.empty(T, CANDIDATES * 8, dtype=torch.float32, device=dev)

                def build():
                    for x, b in zip(xs, btr):
                        pa.build_candidate_gather(x["cand"], x["ends"], b, kv4, H, 128, 8)

                def gathered():
                    for x, (a, bb), b, gth in zip(xs, q4, btr, gath):
                        g, slot_ends = gth
                        pa.paged_mxfp4_mqa_logits(
                            a.view(T, 1, H, 64),
                            bb.view(T, 1, H, 4),
                            kv4,
                            x["w"],
                            x["ends"],
                            b,
                            CANDIDATES * 8,
                            out_logits=compact,
                            clean_logits=False,
                            row_ends=slot_ends,
                            use_gather=True,
                            candidates=g,
                        )
                        vi._topk(compact, slot_ends, topk, TOPK, T)
                        vi._remap_compact_topk(topk, g["positions"], 8)

                graphs = {"build": graph(build), "gather+topk+remap": graph(gathered)}
                if ragged_on:
                    graphs["pack"] = graph(pack)

            def mine():
                for x, r in zip(xs, rg):
                    kw = {"ragged": r} if ragged_on else {}
                    if args.mode == "candidates":
                        ix.indexer_candidate_topk(
                            x["q"],
                            x["qs"],
                            x["w"],
                            pool,
                            x["bt"],
                            x["ends"],
                            x["cand"],
                            out,
                            next_n=nn,
                            max_ctx=bound,
                            **kw,
                        )
                    else:
                        if args.mode == "source":
                            kw["candidates_out"] = cand
                        ix.indexer_topk(
                            x["q"], x["qs"], x["w"], pool, x["bt"], x["ends"], out, next_n=nn, max_ctx=bound, **kw
                        )

            graphs["mine"] = graph(mine)
            t = abba(graphs)
            base = sum(v for k, v in t.items() if k != "mine") - (3 * t["schedule"] / 4 if "schedule" in t else 0)
            kvmb = sum(lengths) * ENTRY_BYTES / 1e6
            detail = ", ".join(f"{k} {v:.1f}" for k, v in t.items() if k != "mine")
            print(
                f"{ctx} | {reqs} | {T} | {kvmb:.0f} | {kvmb / 8:.1f} | {base:.1f} | {t['mine']:.1f} | "
                f"{base / t['mine']:.2f}x | {detail}",
                flush=True,
            )
            xs = graphs = None
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
