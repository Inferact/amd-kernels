# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the indexer kernels on gfx950, index pages HBM-cold.

Timing (``timeit_graph``): one HIP graph
holds ``copies`` calls, each on its own input (index query, block table into its
own pages of a page pool larger than the 256 MB MALL, lengths), so the scorers
read their pages from HBM as in serving; the score buffer and the top-k's input
are shared (producer-warm). The median over 5 rounds of 10 replays, per call.
Before timing, the first and last input are checked against ``reference.py``.

- ``--decode`` (default): the decode chain per layer, the scorer
  (``nvfp4_index_score_decode``) and the indices top-k, at max_seq_len =
  max_model_len (what decode graphs capture);
- ``--prefill``: the prefill scorer and the indices top-k.
"""

import argparse
import math
import os
import statistics
import sys

import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    D,
    SCORE_ATOL,
    SCORE_RTOL,
    cdiv,
    decode_scores_reference,
    nvfp4_index_pages,
    prefill_scores_reference,
    request_keys,
    score_width,
    topk_reference,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import nvfp4_index_score_decode
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_prefill import nvfp4_index_score_prefill
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import (
    index_topk_decode_indices,
    index_topk_prefill_indices,
)

INIT, LOCAL = 0, 1
MAX_MODEL_LEN = 1 << 20
PAGE_BYTES = BLK * 72
POOL_BYTES = 768 << 20  # page pool, above the 256 MB MALL
COPIES = 100


def log(*args):
    if os.environ.get("BENCH_VERBOSE"):
        print(*args, file=sys.stderr, flush=True)


def timeit_graph(calls, replays=10, rounds=5, warmup=3):
    """Median over rounds of the mean time per call (us) of one graph of ``calls``."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / (replays * len(calls)) for s, e in events) * 1e3


def page_pool(pages: int) -> torch.Tensor:
    cache = torch.empty(pages, BLK, 72, dtype=torch.uint8, device="cuda")
    for p0 in range(0, pages, 512):
        n = min(512, pages - p0)
        cache[p0 : p0 + n] = nvfp4_index_pages((torch.randn(n * BLK, D, device="cuda") * 2).bfloat16())
    return cache


def pool_pages_for(lens):
    """Pages of a pool that holds at least two inputs and more than the MALL."""
    return max(POOL_BYTES // PAGE_BYTES, 2 * sum(cdiv(s, BLK) for s in lens) + 2)


def tables(lens, copies, pool_pages):
    """``copies`` block tables: the pool in slots of one input's pages, input i in
    slot i % slots (an input's pages come back after at least 384 MB of others),
    each request's blocks in random order."""
    need = max(1, sum(cdiv(s, BLK) for s in lens))
    slots = pool_pages // need
    cols = max(cdiv(max(lens), BLK), 1)
    out = []
    for i in range(copies):
        base = (i % slots) * need
        perm = base + torch.randperm(need, device="cuda", dtype=torch.int32)
        t = torch.zeros(len(lens), cols, dtype=torch.int32, device="cuda")
        start = 0
        for r, s in enumerate(lens):
            w = cdiv(s, BLK)
            t[r, :w] = perm[start : start + w]
            start += w
        out.append(t)
    return out


def num_copies(page_bytes_per_call):
    return max(COPIES, min(400, math.ceil(POOL_BYTES / max(1, page_bytes_per_call))))


def check_decode(q, cache, score, table, lens, ql):
    want = decode_scores_reference(q, cache, table, lens, INIT, LOCAL, ql)
    for r, s in enumerate(lens):
        got = score[:, r * ql : (r + 1) * ql, : cdiv(s, BLK)]
        torch.testing.assert_close(got.double(), want[r], rtol=SCORE_RTOL, atol=SCORE_ATOL)


def bench_decode(heads, ql, lens, label):
    n = len(lens)
    rows = n * ql
    page_bytes = sum(cdiv(s, BLK) for s in lens) * PAGE_BYTES
    copies = num_copies(page_bytes)
    pool_pages = pool_pages_for(lens)
    cache = page_pool(pool_pages)
    tabs = tables(lens, copies, pool_pages)
    qs = [(torch.randn(rows, heads, D, device="cuda") * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    score = torch.empty(heads, rows, score_width(MAX_MODEL_LEN), device="cuda")
    topk = torch.empty(heads, rows, 16, dtype=torch.int32, device="cuda")

    def scorer(i):
        nvfp4_index_score_decode(qs[i], cache, score, tabs[i], seq, INIT, LOCAL, ql, MAX_MODEL_LEN)

    def topk_call():
        index_topk_decode_indices(score, topk, seq, max_seq_len=MAX_MODEL_LEN, query_len=ql)

    log("decode", heads, ql, n, label, "copies", copies)
    for i in (0, copies - 1):
        score.fill_(float("nan"))
        scorer(i)
        topk_call()
        torch.cuda.synchronize()
        check_decode(qs[i], cache, score, tabs[i], lens, ql)
        kv = seq.long().repeat_interleave(ql) - ql + torch.arange(rows, device="cuda") % ql + 1
        nblk = (kv.clamp(min=0) + BLK - 1) // BLK
        assert torch.equal(topk, topk_reference(score, nblk, MAX_MODEL_LEN))
    log("checked")
    s_us = timeit_graph([lambda i=i: scorer(i) for i in range(copies)])
    log("scorer", s_us)
    c_us = timeit_graph([lambda i=i: (scorer(i), topk_call()) for i in range(copies)])
    # the top-k alone, over the scores of 16 inputs (its time depends on the data)
    scores = [torch.empty_like(score) for _ in range(16)]
    for j, buf in enumerate(scores):
        nvfp4_index_score_decode(qs[j], cache, buf, tabs[j], seq, INIT, LOCAL, ql, MAX_MODEL_LEN)
    log("chain", c_us)
    t_us = timeit_graph(
        [
            lambda b=b: index_topk_decode_indices(b, topk, seq, max_seq_len=MAX_MODEL_LEN, query_len=ql)
            for b in scores
        ]
        * 4
    )
    log("topk", t_us)
    gbs = page_bytes / (s_us * 1e-6) / 1e9
    print(
        f"| h{heads} q{ql} | {n} | {label} | {s_us:.1f} | {gbs:.0f} | {t_us:.1f} "
        f"| {c_us:.1f} |",
        flush=True,
    )


def bench_prefill(heads, q_lens, seq_lens, label):
    total = sum(q_lens)
    msl = max(seq_lens)
    copies = min(COPIES, max(8, (8 << 30) // max(1, total * heads * D)))
    pool_pages = pool_pages_for(seq_lens)
    cache = page_pool(pool_pages)
    tabs = tables(seq_lens, copies, pool_pages)
    qs = [(torch.randn(total, heads, D, device="cuda") * 0.5).to(torch.float8_e4m3fn) for _ in range(copies)]
    seq = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
    cu = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.tensor(q_lens, dtype=torch.int32, device="cuda").cumsum(0)
    score = torch.empty(heads, total, score_width(msl), device="cuda")
    topk = torch.empty(heads, total, 16, dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(s - n, s, device="cuda") for n, s in zip(q_lens, seq_lens)])
    nvp = (pos // BLK + 1).to(torch.int32)

    def scorer(i):
        nvfp4_index_score_prefill(qs[i], cache, score, tabs[i], cu, seq, INIT, LOCAL, max(q_lens), msl)

    def topk_call():
        index_topk_prefill_indices(score, topk, max_seq_len=msl, block_size=BLK, num_valid_pages=nvp)

    for i in (0, copies - 1):
        scorer(i)
        topk_call()
        torch.cuda.synchronize()
        q0 = 0
        for r, (n, s) in enumerate(zip(q_lens, seq_lens)):
            keys = request_keys(cache, tabs[i], r, cdiv(s, BLK), False)
            p = s - n + torch.arange(n, device="cuda")
            for r0 in range(0, n, 1024):  # a few row slices
                r1 = min(n, r0 + 512)
                want, seen = prefill_scores_reference(qs[i][q0 + r0 : q0 + r1], keys, p[r0:r1], INIT, LOCAL)
                got = score[:, q0 + r0 : q0 + r1, : cdiv(s, BLK)]
                sm = seen[None].expand(heads, -1, -1)
                torch.testing.assert_close(got[sm], want[sm], rtol=SCORE_RTOL, atol=SCORE_ATOL)
            q0 += n
        assert torch.equal(topk, topk_reference(score, nvp.long(), msl))
    s_us = timeit_graph([lambda i=i: scorer(i) for i in range(copies)], replays=3)
    scores = [torch.empty_like(score) for _ in range(4)]
    for j, buf in enumerate(scores):
        nvfp4_index_score_prefill(qs[j], cache, buf, tabs[j], cu, seq, INIT, LOCAL, max(q_lens), msl)
    t_us = timeit_graph(
        [
            lambda b=b: index_topk_prefill_indices(
                b, topk, max_seq_len=msl, block_size=BLK, num_valid_pages=nvp
            )
            for b in scores
        ]
        * 4,
        replays=3,
    )
    flops = sum(n * heads * (s - n / 2) * D * 2 for n, s in zip(q_lens, seq_lens))
    print(
        f"| h{heads} | {label} | {s_us:.1f} | {flops / (s_us * 1e-6) / 1e15:.2f} | {t_us:.1f} |",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", action="store_true")
    args = ap.parse_args()
    dev = torch.cuda.get_device_properties(0)
    print(
        f"{dev.name} {dev.gcnArchName}, HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES')}",
        flush=True,
    )
    if args.prefill:
        print("| Heads | New tokens / context | Scorer us | PF/s | Top-k us |")
        print("|---|---|---:|---:|---:|")
        for q_lens, seq_lens, label in [
            ([4096], [4096], "4K / 4K"),
            ([8192], [65536], "8K / 64K"),
            ([16384], [131072], "16K / 128K"),
            ([32768], [32768], "32K / 32K"),
            ([2048], [1 << 20], "2K / 1M"),
            ([3072, 3072, 3072], [203072, 3072, 207872], "3 x 3K / 200K, 3K, 208K"),
        ]:
            bench_prefill(1, q_lens, seq_lens, label)
        return
    print("| Heads x rows | Requests | Lengths | Scorer us | Page GB/s | Top-k us | Chain us |")
    print("|---|---:|---|---:|---:|---:|---:|")
    rng = torch.Generator().manual_seed(0)
    for heads, ql, counts in ((1, 4, (16, 64, 128, 256)),):
        for n in counts:
            for label, lens in (
                ("32K", [32768] * n),
                ("128K", [131072] * n),
                ("30K-210K", torch.randint(30000, 210001, (n,), generator=rng).tolist()),
            ):
                bench_decode(heads, ql, lens, label)


if __name__ == "__main__":
    main()
