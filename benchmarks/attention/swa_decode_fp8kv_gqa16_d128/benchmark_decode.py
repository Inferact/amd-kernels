# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``swa_decode_fp8kv`` on gfx950, KV cold.

One HIP graph holds ``--copies`` (100) launches, each with its own queries, block table
and sequence lengths. Each launch reads its own pages of a KV pool of at least 1 GiB
(more than the 256 MB MALL; consecutive launches take disjoint pages), so the window's
K / V come from HBM as in serving. Every request has a full window (context 8K..64K
tokens; window 4096: 4095 + rows keys). Median of 5 rounds of 10 replays, in µs per
launch; KV GB/s counts the window's K / V bytes once per (request, KV head). The first
and the last launch are checked against ``reference.py`` before timing.

Cases (README, "What it does"): TP4 spec3 (16 / 1 heads, 4 rows, bf16 output,
128-token pages); the TP4 DSpark draft's 7-row block, causal and non-causal (the
upstream vLLM draft). The MXFP8 epilogue with o_proj: ``benchmarks/attention/bench_attn_oproj_mxfp8.py``.
"""

import argparse
import statistics

import torch
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128 import host, swa_decode_fp8kv
from amd_kernels.attention.swa_decode_fp8kv_gqa16_d128.reference import (
    D,
    HG,
    TOK_B,
    check_close,
    decode_reference,
)

W = 4096
K_SCALE, V_SCALE = 0.75, 1.25
SM_SCALE = D**-0.5
POOL_BYTES = 1 << 30
# (id, rows per request, KV heads, page, request counts[, causal (default True)])
CASES = [
    ("tp4_spec3_p128_bf16", 4, 1, 128, [1, 16, 64, 128, 256]),
    ("tp4_dspark7_p128_bf16", 7, 1, 128, [1, 2, 4, 16, 32, 64]),
    ("tp4_dspark7_noncausal_p128_bf16", 7, 1, 128, [1, 2, 4, 8, 11, 16, 24, 32, 48, 64], False),
]


def timeit_graph(calls, inner=1, replays=10, rounds=5, warmup=3):
    """Median over rounds of the mean time per call (ms) of one graph that runs
    ``calls`` ``inner`` times (the router benchmark's ``timeit_graph``)."""
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(len(calls) * inner):
            calls[i % len(calls)]()
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
    return statistics.median(s.elapsed_time(e) / (replays * len(calls) * inner) for s, e in events)


def make_launches(reqs, ql, hkv, page, copies, seed):
    """``copies`` inputs over one pool; returns (pool, [(bt, seq_lens, qsl, q)])."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    per_req = (W + ql + page - 1) // page + 1
    per_call = reqs * per_req
    page_bytes = hkv * page * TOK_B
    n_pages = max(2 * per_call, -(-POOL_BYTES // page_bytes))
    pool = torch.empty((n_pages, hkv, page, TOK_B), dtype=torch.uint8, device="cuda")
    for i in range(0, n_pages, 256):  # e4m3 of N(0, 1.2), in chunks
        chunk = pool[i : i + 256]
        vals = torch.randn(chunk.shape, generator=gen, device="cuda", dtype=torch.bfloat16) * 1.2
        chunk.copy_(vals.to(torch.float8_e4m3fn).view(torch.uint8))
    cpu = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_pages, generator=cpu).int()
    order = torch.cat([order, order])
    launches = []
    cursor = 0
    for _ in range(copies):
        ctx = torch.randint(8192, 65536, (reqs,), generator=cpu)
        seq = ctx + ql
        nblk = int((seq.max() + page - 1) // page)
        bt = torch.zeros((reqs, nblk), dtype=torch.int32)
        for r, (c, s_r) in enumerate(zip(ctx.tolist(), seq.tolist())):
            nb = (s_r + page - 1) // page
            lo = (c - W + 1) // page
            bt[r, lo:nb] = order[cursor : cursor + nb - lo]
            cursor = (cursor + nb - lo) % n_pages
        qsl = torch.arange(0, (reqs + 1) * ql, ql, dtype=torch.int32)
        q = (torch.randn((reqs * ql, HG * hkv, D), generator=gen, device="cuda") * 1.5).bfloat16()
        launches.append((bt.cuda(), seq.int().cuda(), qsl.cuda(), q))
    return pool, launches


def bench(name, ql, hkv, page, reqs, copies, scales, causal=True):
    ks, vs = scales
    pool, launches = make_launches(reqs, ql, hkv, page, copies, seed=reqs * 100 + ql * 10 + hkv)
    rows = reqs * ql
    out = torch.empty((rows, HG * hkv, D), dtype=torch.bfloat16, device="cuda")

    def call(i):
        bt, seq, qsl, q = launches[i]
        swa_decode_fp8kv(q, pool, bt, seq, qsl, ql, SM_SCALE, W, causal, ks, vs, out)

    for i in (0, copies - 1):  # the contract, before timing
        bt, seq, qsl, q = launches[i]
        swa_decode_fp8kv(q, pool, bt, seq, qsl, ql, SM_SCALE, W, causal, ks, vs, out)
        ref = decode_reference(q, pool, bt, seq, qsl, SM_SCALE, W, causal, K_SCALE, V_SCALE)
        for h in range(hkv):
            sl = (slice(None), slice(HG * h, HG * (h + 1)))
            check_close(out[sl], ref[sl], f"{name} R={reqs} launch {i} KV head {h}")
    us = timeit_graph([lambda i=i: call(i) for i in range(copies)]) * 1e3
    kv_bytes = reqs * hkv * (W - 1 + ql) * TOK_B
    splits = host.launch_splits(launches[0][3], pool, reqs)
    print(
        f"{name:22s} R={reqs:3d} rows={rows:4d} splits={splits}: "
        f"{us:7.2f} us, {kv_bytes / us / 1e3:6.0f} GB/s KV"
    )
    return us


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=100, help="launches (inputs) per graph")
    parser.add_argument("--case", nargs="*", default=[c[0] for c in CASES])
    args = parser.parse_args()
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("requires gfx950")
    scales = torch.tensor(K_SCALE, device="cuda"), torch.tensor(V_SCALE, device="cuda")
    with torch.inference_mode():
        for name, ql, hkv, page, reqs_list, *causal in CASES:
            if name in args.case:
                for reqs in reqs_list:
                    bench(name, ql, hkv, page, reqs, args.copies, scales, *causal)


if __name__ == "__main__":
    main()
