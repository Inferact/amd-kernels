# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``fp8_dense_mla_decode`` against AITER's ``mla_decode_fwd`` (the stock draft decode) on gfx950.

Served configs of the Kimi-K3.1 DSpark draft under DCP8: c16 (8 requests x 4 rows), c24 (13 x 4), c32 (20 x 3),
c48 (48 x 3), each over ``--local-len`` keys per request on this rank (a context of 8 x local-len). One HIP graph
holds ``--copies`` launches over disjoint KV caches (the cache of all copies is larger than the 256 MB MALL), so the keys
come from HBM as in serving. Median and minimum of ``--rounds`` replays, in us per launch. Both kernels see the same cache,
queries, scales and key lists; the stock path is run as vLLM runs it (96 heads padded to 128, persistent-schedule metadata
built once outside the graph, ``cprr`` round-robin causal kernel plus its reduce).

The roofline columns: FLOPs = 2 * (576 + 512) * rows * 96 * keys (the work of the real heads, not the padding) against
``--mfma-pf`` (the fp8 MFMA rate under this kernel's load, README) and the KV bytes against ``--hbm-tbs``.
"""

import argparse
import statistics

import torch
from amd_kernels.attention.fp8_dense_mla_decode_h96_kv512 import fp8_dense_mla_decode, supports_fp8_dense_mla_decode
from amd_kernels.attention.fp8_dense_mla_decode_h96_kv512.reference import fp8_dense_mla_decode_ref

CASES = [("c16", 8, 4), ("c24", 13, 4), ("c32", 20, 3), ("c48", 48, 3)]
Q_SCALE, KV_SCALE = 0.5, 0.25
SM_SCALE = 192**-0.5
FP8 = torch.float8_e4m3fn
W = 8


def make_inputs(reqs, qlen, local_len, copies, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = reqs * local_len
    blocks = -(-total // 64) + 2
    sets = []
    for c in range(copies):
        cache = torch.randn((blocks * 64, 576), generator=gen, device="cuda").to(FP8)
        order = torch.randperm(blocks, generator=gen, device="cuda").to(torch.int32)
        idx = (order[:, None] * 64 + torch.arange(64, device="cuda", dtype=torch.int32)[None, :]).flatten()[:total].contiguous()
        q = torch.randn((reqs * qlen, 96, 576), generator=gen, device="cuda").to(FP8)
        ip = torch.arange(0, (reqs + 1) * local_len, local_len, dtype=torch.int32, device="cuda")
        gip = ((ip[1:] - ip[:-1]) * W).cumsum(0).to(torch.int32)
        gip = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), gip])
        sets.append((q, cache, ip, idx, gip))
    return sets


def stock_runner(sets, reqs, qlen):
    from aiter import dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
    from aiter.mla import mla_decode_fwd

    runs = []
    qs = torch.tensor([Q_SCALE], dtype=torch.float32, device="cuda")
    ks = torch.tensor([KV_SCALE], dtype=torch.float32, device="cuda")
    for q, cache, ip, idx, gip in sets:
        qp = q.repeat(1, 2, 1)[:, :128].contiguous()
        qo = torch.arange(0, (reqs + 1) * qlen, qlen, dtype=torch.int32, device="cuda")
        last = torch.ones(reqs, dtype=torch.int32, device="cuda")
        info = get_mla_metadata_info_v1(reqs, qlen, 128, dtypes.fp8, dtypes.fp8, is_sparse=False, fast_mode=True, max_split_per_batch=256)
        wm, wip, wis, rip, rfm, rpm = [torch.empty(s, dtype=t, device="cuda") for (s, t) in info]
        get_mla_metadata_v1(qo, ip, last, 128, 1, False, wm, wis, wip, rip, rfm, rpm, page_size=1, kv_granularity=16,
                            max_seqlen_qo=qlen, uni_seqlen_qo=qlen, fast_mode=True, dtype_q=dtypes.fp8, dtype_kv=dtypes.fp8,
                            max_split_per_batch=256, intra_batch_mode=False, is_cp_round_robin=True)
        o = torch.empty(reqs * qlen, 128, 512, dtype=torch.bfloat16, device="cuda")
        kv = cache.view(-1, 1, 1, 576)

        def run(qp=qp, kv=kv, o=o, qo=qo, ip=ip, idx=idx, last=last, gip=gip, wm=wm, wip=wip, wis=wis, rip=rip, rfm=rfm, rpm=rpm):
            mla_decode_fwd(qp, kv, o, qo, ip, idx, last, qlen, sm_scale=SM_SCALE, return_lse=True, causal=True, work_meta_data=wm,
                           work_indptr=wip, work_info_set=wis, reduce_indptr=rip, reduce_final_map=rfm, reduce_partial_map=rpm,
                           q_scale=qs, kv_scale=ks, g_kv_indptr=gip, cp_world_size=W, cp_rank=0, num_kv_splits=256,
                           intra_batch_mode=False)

        runs.append(run)
    return runs


def time_graph(calls, rounds, replays=5):
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    graph.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rounds):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(replays):
            graph.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (replays * len(calls)))
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-len", type=int, default=15000)
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--mfma-pf", type=float, default=4.07)
    ap.add_argument("--hbm-tbs", type=float, default=6.5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cases", default=",".join(c[0] for c in CASES))
    args = ap.parse_args()
    qs = torch.tensor([Q_SCALE], dtype=torch.float32, device="cuda")
    ks = torch.tensor([KV_SCALE], dtype=torch.float32, device="cuda")
    print(f"local keys per request {args.local_len}; us per launch (median / min of {args.rounds} replays)")
    print("case  reqs rows | stock us | kernel us | speedup | TF/s | %MFMA | KV GB/s | %HBM")
    for name, reqs, qlen in CASES:
        if name not in args.cases.split(","):
            continue
        sets = make_inputs(reqs, qlen, args.local_len, args.copies, seed=reqs)
        assert all(supports_fp8_dense_mla_decode(q, c, ip, idx, gip, qs, ks, qlen) for q, c, ip, idx, gip in sets)
        if args.check:
            q, c, ip, idx, gip = sets[0]
            out, lse = fp8_dense_mla_decode(q, c, ip, idx, gip, qs, ks, SM_SCALE, qlen)
            ro, rl = fp8_dense_mla_decode_ref(q, c, ip, idx, gip, Q_SCALE, KV_SCALE, SM_SCALE, qlen, W, 0)
            cos = torch.nn.functional.cosine_similarity(out.float(), ro.float(), dim=-1).min().item()
            print(f"  check {name}: min cosine {cos:.5f}, max |lse err| {(lse - rl).abs().max().item():.1e}")
        mine = [lambda s=s: fp8_dense_mla_decode(s[0], s[1], s[2], s[3], s[4], qs, ks, SM_SCALE, qlen) for s in sets]
        t_mine = time_graph(mine, args.rounds)
        t_stock = time_graph(stock_runner(sets, reqs, qlen), args.rounds)
        flops = 2 * 1088 * reqs * qlen * 96 * args.local_len
        kv_bytes = reqs * args.local_len * 576
        tf = flops / t_mine[1] / 1e6
        gb = kv_bytes / t_mine[1] / 1e3
        print(
            f"{name:5s} {reqs:4d} {qlen:4d} | {t_stock[0]:8.1f} / {t_stock[1]:8.1f} | {t_mine[0]:8.1f} / {t_mine[1]:8.1f} | "
            f"{t_stock[1] / t_mine[1]:5.2f}x | {tf:7.0f} | {100 * tf / (args.mfma_pf * 1e3):5.1f} | {gb:7.0f} | {100 * gb / (args.hbm_tbs * 1e3):5.1f}"
        )


if __name__ == "__main__":
    main()
