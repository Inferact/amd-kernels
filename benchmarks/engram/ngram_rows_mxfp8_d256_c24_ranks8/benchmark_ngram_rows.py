# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay time of ``NgramRows`` (hash + both layers' row pull) against vLLM's Triton chain (``_hash_ids_kernel``
+ one ``_engram_lookup_kernel`` per layer, BF16 rows, the peer-pointer table of its ``peer_rows`` path), per row count
of a decode step (requests of 6 rows, 160 request slots, padding rows). One GPU: the eight shards are one allocation
at the served size (48000771 rows x 264 B per layer, 12.7 GB), so every pull is a local-HBM random read (no xGMI).
100 distinct batches per graph, median of 5 rounds x 5 replays, us per call.
Run: python3 benchmarks/engram/ngram_rows_mxfp8_d256_c24_ranks8/benchmark_ngram_rows.py [--rows 16 64 ...]
"""

import argparse
import statistics

import torch
from amd_kernels.engram.ngram_rows_mxfp8_d256_c24_ranks8 import NgramRows
from tests.engram.ngram_rows_mxfp8_d256_c24_ranks8 import workload as wl

NIN = 100
SLOTS = 160


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


def vllm_chain(tmap, mult, primes, offsets, pad_id, peer_ptrs, peer_rows, totals):
    from vllm.models.deepseek_v41.common import engram as ve
    from vllm.triton_utils import triton

    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def run(ids, pos, qsl, lb, hashes, rows):
        m = ids.shape[0]
        dead = ids == wl.IMAGE_ID
        lb_dead = lb == wl.IMAGE_ID
        ve._hash_ids_kernel[(triton.cdiv(m, 32), 2)](
            ids, tmap, dead, pos, qsl, mult, primes.view(2, 3, 8), offsets.view(2, 3, 8), lb, lb_dead, hashes, m, pad_id,
            ids.stride(0), dead.stride(0), pos.stride(0), qsl.stride(0), qsl.numel() - 1, MAX_NGRAM=4, num_heads=8,
            BLOCK_T=32, BLOCK_H=8, dead_id=-1, lookback_depth=3, lookback_row_stride=lb.stride(0),
            lookback_col_stride=lb.stride(1), lookback_mask_row_stride=lb_dead.stride(0),
            lookback_mask_col_stride=lb_dead.stride(1), num_warps=4)
        n = m * 24
        block_r = 128 if n >= 128 * sms else 16
        grid = min(triton.cdiv(n, block_r), sms)
        for layer in range(2):
            ve._engram_lookup_kernel[(grid,)](
                peer_ptrs[layer], peer_ptrs[layer], hashes[:, layer], hashes, rows[layer], 0, totals[layer], n, hashes.stride(0), 1,
                HEAD_START=0, LOCAL_HEADS=24, TOTAL_HEADS=24, DIM=256, QUANT_BLOCK=32, BLOCK_R=block_r, GRID=grid,
                SORTED=False, PACKED=False, OUT_STRIDE=256, PEER_ROWS=peer_rows, num_warps=8 if block_r == 128 else 4)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64, 128, 192, 256, 384, 512, 768])
    args = ap.parse_args()
    dev = torch.device("cuda")
    primes, offsets, mult = wl.layout()
    totals = [int(primes[layer].sum()) for layer in range(2)]
    peer_rows = -(-max(totals) // wl.RANKS)
    tmap = wl.token_map(3).to(dev)
    pad_id = int(tmap[wl.PAD_ID_TOKEN])
    bufs = [torch.empty(peer_rows * (256 + 8), dtype=torch.uint8, device=dev) for _ in range(2)]
    for b in bufs:
        b.view(torch.int64).random_()
    wptr = [[b.data_ptr()] * wl.RANKS for b in bufs]
    sptr = [[b.data_ptr() + peer_rows * 256] * wl.RANKS for b in bufs]
    peer_ptrs = [torch.tensor([b.data_ptr()] * wl.RANKS, dtype=torch.int64, device=dev) for b in bufs]
    mult_d, primes_d, offsets_d = mult.to(dev), primes.to(dev), offsets.to(dev)
    ours = NgramRows(tmap, mult_d, primes_d, offsets_d, pad_id, peer_rows, wptr, sptr, max_rows=max(args.rows),
                     image_id=wl.IMAGE_ID, keepalive=bufs)
    ours.warm_up()
    try:
        chain = vllm_chain(tmap, mult_d, primes_d.view(-1), offsets_d.view(-1), pad_id, peer_ptrs, peer_rows, totals)
    except ImportError:
        chain = None
    print("rows | this entry us | vLLM Triton chain us")
    for m in args.rows:
        ins = []
        for i in range(NIN):
            ids, pos, qsl, lb, _ = wl.decode_batch(max(1, m // 6), 6, m, 1000 * m + i, slots=SLOTS)
            ins.append([t.to(dev) for t in (ids, pos, qsl, lb)])

        def ours_step():
            for ids, pos, qsl, lb in ins:
                ours(ids, pos, qsl, lb)

        g = graph(ours_step)
        t_ours = statistics.median([timed(g) for _ in range(2)])
        t_base = float("nan")
        if chain is not None:
            hashes = torch.empty(m, 2, 24, dtype=torch.int32, device=dev)
            rows = [torch.empty(m, 24, 256, dtype=torch.bfloat16, device=dev) for _ in range(2)]

            def base_step():
                for ids, pos, qsl, lb in ins:
                    chain(ids, pos, qsl, lb, hashes, rows)

            gb = graph(base_step)
            t_base = statistics.median([timed(gb) for _ in range(2)])
            del gb
        print(f"{m} | {t_ours:.1f} | {t_base:.1f}", flush=True)
        del g, ins


if __name__ == "__main__":
    main()
