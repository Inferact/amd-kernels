# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``qkv_a_norm_kv_insert`` and ``indexer_prep`` against their skeletons (the same memory
dependence chain without arithmetic, ``skeleton.py``: the empirical floor) and against the unfused ops vLLM runs
(the multi-stream DSA wrapper's chain: AITER ``rms_norm`` x2 + ``concat_and_cache_mla``; ``layer_norm``, Triton
``store_keys``, ``fill_``, the fp32 weight product, the starts clamp and ``quantize_queries``; also the two-launch
Triton ``store_keys_and_reset`` + ``quantize_queries_and_weights`` of ``VLLM_K3_FUSED_INDEXER_PREP``).

One HIP graph holds ``--copies`` launches with their own inputs and disjoint cache slots, so the inputs of all copies
exceed the 256 MB MALL together and each launch reads from HBM. Minimum of 3 medians of 5 rounds of 10 replays, us per
call. The first launch of each variant is checked against ``reference.py`` before timing. "% of ceiling" is the
skeleton's time over the kernel's. Bytes are what a call reads and writes; a bare launch takes about 1.6 us in a graph.
"""

import argparse
import statistics

import torch
import torch.nn.functional as F
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168 import reference as R
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import indexer_prep
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import (
    IndexSide,
    qkv_a_norm_kv_insert,
)
from skeleton import skeleton_indexer, skeleton_qkv_a

EPS = 1e-6
TOPK = 2048
HBM_TBS = 8.0
ACHIEVABLE = 0.8
SERVED_M = [6, 12, 24, 40, 65, 80, 128, 192]
WEIGHT_SCALE = 32**-0.5 * 128**-0.5
PAGE_STRIDE = 41984  # the merged MLA page: 64 x 656


def timeit_graph(calls, replays=10, rounds=5, warmup=3, repeat=3):
    return min(_timeit_graph(calls, replays, rounds, warmup) for _ in range(repeat))


def _timeit_graph(calls, replays, rounds, warmup):
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        graph.replay()
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / (replays * len(calls)) * 1000 for s, e in events)


def make_qkv(m, copies, gen, width=2112):
    dev = "cuda"
    xs = [(torch.randn((m, width), generator=gen, device=dev) * 2).bfloat16() for _ in range(copies)]
    qw = (torch.randn(1536, generator=gen, device=dev) * 0.3 + 1).bfloat16()
    kw = (torch.randn(512, generator=gen, device=dev) * 0.3 + 1).bfloat16()
    pages = copies * ((m + 63) // 64) + 1
    base = torch.zeros(pages * PAGE_STRIDE, dtype=torch.uint8, device=dev)
    cache = base.as_strided((pages, 64, 656), (PAGE_STRIDE, 656, 1))
    slots = [torch.arange(m, device=dev, dtype=torch.int64) + i * ((m + 63) // 64) * 64 for i in range(copies)]
    qo = [torch.empty((m, 1536), dtype=torch.bfloat16, device=dev) for _ in range(copies)]
    ko = [torch.empty((m, 512), dtype=torch.bfloat16, device=dev) for _ in range(copies)]
    return xs, qw, kw, cache, slots, qo, ko


def make_index(m, copies, gen, rows_out):
    dev = "cuda"
    mk = lambda *s: (torch.randn(s, generator=gen, device=dev) * 2).bfloat16()  # noqa: E731
    qs_ = [mk(m, 32, 128) for _ in range(copies)]
    ks = [mk(m, 128) for _ in range(copies)]
    ws = [mk(m, 32) for _ in range(copies)]
    lw = (torch.randn(128, generator=gen, device=dev) * 0.3 + 1).bfloat16()
    lb = (torch.randn(128, generator=gen, device=dev) * 0.1).bfloat16()
    pages = copies * ((m + 63) // 64) + 1
    base = torch.zeros(pages * PAGE_STRIDE, dtype=torch.uint8, device=dev)
    cache = base.as_strided((pages, 64, 68), (PAGE_STRIDE, 68, 1))
    slots = [torch.arange(m, device=dev, dtype=torch.int64) + i * ((m + 63) // 64) * 64 for i in range(copies)]
    outs = [torch.zeros((rows_out, TOPK), dtype=torch.int32, device=dev) for _ in range(copies)]
    qp = [torch.empty((m, 32, 64), dtype=torch.uint8, device=dev) for _ in range(copies)]
    qsc = [torch.empty((m, 1, 4, 16, 4), dtype=torch.uint8, device=dev) for _ in range(copies)]
    wo = [torch.empty((m, 32), dtype=torch.float32, device=dev) for _ in range(copies)]
    return qs_, ks, ws, lw, lb, base, cache, slots, outs, qp, qsc, wo


def bench_qkv(m, copies, gen):
    from vllm import _custom_ops as ops
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    import aiter

    xs, qw, kw, cache, slots, qo, ko = make_qkv(m, copies, gen)
    one = torch.ones(1, device="cuda")
    each = range(copies)
    ours = [(lambda i=i: qkv_a_norm_kv_insert(xs[i], qw, kw, EPS, qo[i], ko[i], cache, slots[i])) for i in each]
    skel = [(lambda i=i: skeleton_qkv_a(xs[i], qw, kw, qo[i], ko[i], cache, slots[i])) for i in each]

    def stock(i):
        aiter.rms_norm(xs[i][:, :1536], qw, EPS)
        kv_c = aiter.rms_norm(xs[i][:, 1536:2048], kw, EPS)
        ops.concat_and_cache_mla(
            kv_c, xs[i][:, 2048:2112], cache, slots[i], kv_cache_dtype="fp8_ds_mla", scale=one, skip_scale=False
        )

    def stock_fused_norm(i):
        _, kv_c = fused_q_kv_rmsnorm(xs[i][:, :1536], xs[i][:, 1536:2048], qw, kw, EPS)
        ops.concat_and_cache_mla(
            kv_c, xs[i][:, 2048:2112], cache, slots[i], kv_cache_dtype="fp8_ds_mla", scale=one, skip_scale=False
        )

    before = cache.clone()
    ours[0]()
    rq, rkv, rc, wm = R.qkv_a_norm_kv_insert_reference(xs[0], qw, kw, EPS, before, slots[0])
    R.check_norm(qo[0], rq, "q")
    R.check_norm(ko[0], rkv, "kv")
    R.check_kv_cache(cache, rc, before, wm, "cache")
    byt = m * (2112 * 2 + 2048 * 2 + 656) + 1536 * 2 + 512 * 2
    return dict(
        ours=timeit_graph(ours), skel=timeit_graph(skel),
        stock=timeit_graph([(lambda i=i: stock(i)) for i in each]),
        fused=timeit_graph([(lambda i=i: stock_fused_norm(i)) for i in each]), bytes=byt,
    )  # fmt: skip


def bench_qkv_index(m, copies, gen):
    """The row carries the indexer key and head weights: the key path leaves the aux stream and the wk / wv GEMMs merge into qkv_a's."""
    xs, qw, kw, cache, slots, qo, ko = make_qkv(m, copies, gen, 2272)
    lw = torch.ones(128, device="cuda").bfloat16()
    lb = torch.zeros(128, device="cuda").bfloat16()
    pages = copies * ((m + 63) // 64) + 1
    ibase = torch.zeros(pages * PAGE_STRIDE, dtype=torch.uint8, device="cuda")
    icache = ibase.as_strided((pages, 64, 68), (PAGE_STRIDE, 68, 1))
    wts = [torch.empty((m, 32), dtype=torch.float32, device="cuda") for _ in range(copies)]
    each = range(copies)
    sides = [IndexSide(lw, lb, icache, slots[i], wts[i], WEIGHT_SCALE) for i in each]
    ours = [(lambda i=i: qkv_a_norm_kv_insert(xs[i], qw, kw, EPS, qo[i], ko[i], cache, slots[i], sides[i])) for i in each]
    skel = [(lambda i=i: skeleton_qkv_a(xs[i], qw, kw, qo[i], ko[i], cache, slots[i], wts[i], icache)) for i in each]
    ours[0]()
    zeros = torch.zeros_like(ibase).as_strided(icache.shape, icache.stride())
    k0 = xs[0][:, 2112:2240].contiguous()
    w0 = xs[0][:, 2240:2272].contiguous()
    *_, rc, written = R.indexer_prep_reference(k0, lw, lb, EPS, slots[0], torch.zeros((m, 32, 128)).bfloat16().cuda(),
                                               w0, WEIGHT_SCALE, torch.zeros((m, 2048), dtype=torch.int32).cuda(), zeros)  # fmt: skip
    R.check_index(icache, rc, zeros, written, "index cache")
    byt = m * (2272 * 2 + 2048 * 2 + 656 + 68 + 128) + 1536 * 2 + 512 * 2
    return dict(ours=timeit_graph(ours), skel=timeit_graph(skel), bytes=byt)


def bench_index(m, copies, gen, queries_only=False):
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        quantize_queries,
        quantize_queries_and_weights,
        store_keys,
        store_keys_and_reset,
    )

    qs_, ks, ws, lw, lb, base, cache, slots, outs, qp, qsc, wo = make_index(m, copies, gen, m)
    starts = torch.arange(m + 1, device="cuda", dtype=torch.int32)
    lw32, lb32 = lw.float(), lb.float()
    each = range(copies)
    kk = (lambda x, i: None) if queries_only else (lambda x, i: x[i])  # noqa: E731
    ours = [
        (
            lambda i=i: indexer_prep(
                qs_[i], kk(ks, i), kk(ws, i), lw, lb, EPS, None if queries_only else slots[i], cache, outs[i],
                WEIGHT_SCALE, qp[i], qsc[i], None if queries_only else wo[i],
            )
        )
        for i in each
    ]  # fmt: skip
    skel = [
        (lambda i=i: skeleton_indexer(qs_[i], kk(ks, i), kk(ws, i), slots[i], cache, qp[i], qsc[i], wo[i], outs[i]))
        for i in each
    ]

    def stock(i):
        ik = F.layer_norm(ks[i], (128,), lw, lb, EPS)
        store_keys(ik, cache, slots[i])
        outs[i].fill_(-1)
        weights = ws[i].float() * WEIGHT_SCALE
        rel = starts.clamp(0, m)
        quantize_queries(qs_[i])
        return weights, rel

    def stock_fused_prep(i):
        store_keys_and_reset(ks[i], cache, slots[i], outs[i], lw32, lb32, EPS)
        quantize_queries_and_weights(qs_[i], ws[i], WEIGHT_SCALE, starts, 0, m)

    before = base.clone()
    ours[0]()
    bc = before.as_strided(cache.shape, cache.stride())
    rp, rs, rw, rt, rc, wm = R.indexer_prep_reference(ks[0], lw, lb, EPS, slots[0], qs_[0], ws[0], WEIGHT_SCALE, outs[0], bc)
    assert torch.equal(qp[0], rp) and torch.equal(qsc[0], rs) and bool((outs[0] == -1).all())
    if not queries_only:
        assert torch.equal(wo[0], rw)
        R.check_index(cache, rc, bc, wm, "cache")
    byt = m * (8192 + 256 + 64 + 2048 + 256 + 128 + 68 + 8192) + 3 * 128 * 2
    if queries_only:
        byt = m * (8192 + 2048 + 256 + 8192)
    out = dict(ours=timeit_graph(ours), skel=timeit_graph(skel), bytes=byt)
    if not queries_only:
        out["stock"] = timeit_graph([(lambda i=i: stock(i)) for i in each])
        out["fused"] = timeit_graph([(lambda i=i: stock_fused_prep(i)) for i in each])
    return out


def row(name, m, r, launches=None):
    hbm = r["bytes"] / (HBM_TBS * ACHIEVABLE * 1e6)
    s = (
        f"{name:20s} {m:4d} | {r['bytes'] / 1e6:5.2f} MB ({hbm:4.2f} us at 6.4 TB/s) | ours {r['ours']:5.2f} us"
        f" | skeleton {r['skel']:5.2f} us = {100 * r['skel'] / r['ours']:3.0f}% of ceiling"
    )
    if "stock" in r:
        s += f" | stock {r['stock']:5.2f} us ({launches[0]} launches, {r['stock'] / r['ours']:4.2f}x)"
        s += f" | stock-fused {r['fused']:5.2f} us ({launches[1]} launches, {r['fused'] / r['ours']:4.2f}x)"
    print(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=100)
    ap.add_argument("--rows", type=int, nargs="*", default=SERVED_M)
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    for m in args.rows:
        row("qkv_a_norm_kv_ins", m, bench_qkv(m, args.copies, gen), (3, 2))
    for m in args.rows:
        row("qkv_a + index side", m, bench_qkv_index(m, args.copies, gen))
    for m in args.rows:
        row("indexer_prep", m, bench_index(m, args.copies, gen), (6, 2))
    for m in args.rows:
        row("indexer_prep queries", m, bench_index(m, args.copies, gen, True))


if __name__ == "__main__":
    main()
