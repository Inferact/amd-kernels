# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of the fused MXFP4 paged scorer + top-2048 against the unfused pair vLLM runs (AITER's row-group
scorer writing fp32 logits ``[rows, width]``, then AITER's per-row top-2048 with scores), per decode batch and width.

Every shape runs ``NIN`` distinct inputs (query rows and a key cache whose pages the inputs pick at random from a pool
larger than the last-level cache) in one graph; minimum of 15 rounds x 5 replays, us per call. The roofline columns
count the index-cache bytes the call must read (68 B a key, once a sequence: its rows share the keys), the query,
weight and output bytes at the measured 6.4 TB/s HBM rate, the fp4 MFMA work (rows x keys x 32 heads x 128 x 2 flop at
the measured 8.24 PFLOP/s), and the launch floor of the design: one graph kernel (1.57 us), one cross-workgroup
flag (0.8 us) and two cold dependent loads (245 ns). "ceiling" = max(bytes, flops) time + that floor; the last column is ceiling / measured.
Run: python3 benchmarks/indexer/mxfp4_paged_score_topk_d128_h32_top2048/benchmark_score_topk.py [--quick]
"""

import argparse

import torch
from amd_kernels.indexer.mxfp4_paged_score_topk_d128_h32_top2048 import host
from amd_kernels.indexer.mxfp4_paged_score_topk_d128_h32_top2048.reference import TOPK

NIN = 24
HBM_TBS = 6.4
FP4_PFLOPS = 8.24
LAUNCH_FLOOR_US = 1.57 + 0.8 + 2 * 0.245  # one graph launch, one cross-workgroup flag, two cold dependent loads
SHAPES = [  # (sequences, rows a sequence, mean key count a sequence per rank)
    (3, 4, 8192),
    (3, 4, 32768),
    (3, 4, 94208),
    (10, 4, 32768),
    (8, 5, 16384),
    (13, 5, 32768),
    (20, 4, 32768),
    (48, 4, 8192),
    (48, 4, 32768),
    (8, 4, 94208),
    (13, 5, 65536),
    (32, 4, 8192),
    (48, 4, 16384),
    (2, 6, 15360),
    (2, 6, 20480),
    (2, 6, 94208),
]


def make_set(seqs, qlen, width, pool, gen):
    """One input: queries, and a block table over ``pool`` pages (random pages of a large shared cache)."""
    rows = seqs * qlen
    blocks = width // 64
    table = torch.randint(0, pool.shape[0], (seqs, blocks), dtype=torch.int32, device="cuda", generator=gen)
    q_payload = torch.randint(0, 256, (rows, 32, 64), dtype=torch.uint8, device="cuda", generator=gen)
    q_scales = torch.zeros(rows, 1, 4, 16, 4, dtype=torch.uint8, device="cuda")
    q_scales[..., :2] = torch.randint(118, 124, (rows, 1, 4, 16, 2), dtype=torch.uint8, device="cuda", generator=gen)
    weights = torch.randn(rows, 32, device="cuda", generator=gen) * 0.2
    lengths = torch.randint(width * 3 // 4, width + 1, (seqs,), device="cuda", generator=gen)
    r = torch.arange(qlen, device="cuda")
    ends = (lengths[:, None] - (qlen - 1 - r)[None, :]).reshape(-1).int()
    return dict(
        q_payload=q_payload, q_scales=q_scales, weights=weights, cache=pool, block_table=table,
        query_start_loc=torch.arange(seqs + 1, dtype=torch.int32, device="cuda") * qlen, row_ends=ends,
        max_query_len=qlen,
    ), int(lengths.sum())


def timed(g, n):
    ts = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / (5 * n))
    return min(ts)


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


def stock_calls(inputs, out_idx, out_val):
    """AITER's scorer into fp32 logits, then its per-row top-k with scores: what vLLM's decode indexer runs."""
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_rowgroup import (
        flydsl_pa_mqa_logits_fp4_rowgroup,
        make_fp4_mqa_plan,
    )
    from aiter.ops.topk import top_k_per_row_decode

    payload, scales = host.cache_views(inputs["cache"])
    rows = inputs["q_payload"].shape[0]
    width = inputs["block_table"].shape[1] * 64
    plan = make_fp4_mqa_plan(
        num_seqs=inputs["query_start_loc"].numel() - 1, max_qlen=inputs["max_query_len"], num_rows=rows, heads=32,
        page_size=64, max_seq_len=width,
    )
    logits = torch.empty(rows, width, dtype=torch.float32, device="cuda")

    def call():
        flydsl_pa_mqa_logits_fp4_rowgroup(
            plan, inputs["q_payload"], inputs["q_scales"], payload, scales, inputs["block_table"], inputs["weights"],
            inputs["query_start_loc"], inputs["row_ends"], out=logits,
        )
        top_k_per_row_decode(logits, 1, inputs["row_ends"], out_idx, rows, *logits.stride(), TOPK, values=out_val)

    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="three shapes (first of the list, the 2 x 6 ones)")
    args = ap.parse_args()
    shapes = SHAPES[-3:] if args.quick else SHAPES
    gen = torch.Generator(device="cuda").manual_seed(0)
    pool_pages = 3 * 1024 * 1024 * 1024 // 4352  # 3 GiB of index pages: far past the last-level cache
    pool = torch.randint(0, 256, (pool_pages, 4352), dtype=torch.uint8, device="cuda", generator=gen)
    pool[:, 4096:] = torch.randint(118, 124, (pool_pages, 256), dtype=torch.uint8, device="cuda", generator=gen)
    pool = pool.unflatten(1, (64, 68))
    print("seqs x rows | width | stock us | fused us | speedup | key MB | TB/s | HBM us | fp4 us | ceiling us | % of ceiling")
    for position, (seqs, qlen, width) in enumerate([shapes[0]] + list(shapes)):
        warm = position == 0  # the first shape runs once more first: the GPU's clocks and caches settle
        sets, keys = zip(*(make_set(seqs, qlen, width, pool, gen) for _ in range(NIN)))
        rows = seqs * qlen
        out_idx = torch.empty(rows, TOPK, dtype=torch.int32, device="cuda")
        out_val = torch.empty(rows, TOPK, dtype=torch.float32, device="cuda")
        stock = [stock_calls(s, out_idx, out_val) for s in sets]
        host_calls = [lambda s=s: host._score_topk(**s, out_indices=out_idx, out_scores=out_val, uniform=True) for s in sets]
        t_stock = timed(graph(lambda: [c() for c in stock]), NIN)
        t_fused = timed(graph(lambda: [c() for c in host_calls]), NIN)
        if warm:
            continue
        mean_keys = sum(keys) / NIN
        read = mean_keys * 68 + rows * (2048 + 256 + 128)
        written = rows * TOPK * 8
        total = read + written
        tbs = total / (t_fused * 1e-6) / 1e12
        hbm = total / (HBM_TBS * 1e12) * 1e6
        fp4 = mean_keys * qlen * 32 * 128 * 2 / (FP4_PFLOPS * 1e15) * 1e6
        ceiling = max(hbm, fp4) + LAUNCH_FLOOR_US
        print(
            f"{seqs:3d} x {qlen} | {width:6d} | {t_stock:7.1f} | {t_fused:7.1f} | {t_stock / t_fused:5.2f}x | "
            f"{total / 1e6:6.1f} | {tbs:5.2f} | {hbm:5.1f} | {fp4:5.1f} | {ceiling:6.1f} | {100 * ceiling / t_fused:5.1f}%",
            flush=True,
        )


if __name__ == "__main__":
    main()
