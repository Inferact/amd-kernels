# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One-GPU timing of the decode block (index top-16 + sparse decode attention) on its two one-GPU routes, to place
``host.SELECT_MAX_ROWS``: ``fused`` (``nvfp4_index_score_rows``, then the attention with ``index_score``: the
in-kernel top-16) and ``chain`` (``nvfp4_index_score_top16``, then the attention), at 1 (TP4) or 2 (TP2) KV / index
heads, fp16 compute, the MXFP8 o_proj input.

Request i of N at the (i + 0.5) / N quantile of a log-uniform 32K-192K context; every call has its own queries and
block tables onto random slots of a pool of index and KV pages (> 1 GB, beyond the MALL), graphs captured at
max_model_len 1M. Sides alternate per round; median us per call. The sides' ids and output bytes are compared first.

    python3 benchmarks/attention/nvfp4_sparse_decode_gqa16_d128_p128_top16/benchmark_index_decode.py --hkv 2 \
        --ql 8 --reqs 1 4 16 32 48
"""

import argparse
import statistics

import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import reference as iref
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_rows,
    nvfp4_index_score_top16,
)

BLK, D = 128, 128
MAX_MODEL_LEN = 1 << 20
POOL_SLOTS = 1 << 17
INIT, LOCAL = 0, 1


def lens_for(n, lo=32768, hi=196608):
    return [round(lo * (hi / lo) ** ((i + 0.5) / n)) for i in range(n)]


def make_pool(hkv, gen):
    keys = (torch.randn(4096 * BLK, D, device="cuda", generator=gen) * 2).bfloat16()
    pages = iref.nvfp4_index_pages(keys)
    icache = pages.repeat(POOL_SLOTS // pages.shape[0], 1, 1)
    kv = torch.randint(0, 256, (POOL_SLOTS, 2 * hkv, BLK, 72), dtype=torch.uint8, device="cuda", generator=gen) & 0x77
    return icache, kv


def make_calls(hkv, ql, reqs, calls, gen):
    lens = lens_for(reqs)
    rows = reqs * ql
    seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
    out = []
    for _ in range(calls):
        bt = torch.zeros(reqs, MAX_MODEL_LEN // BLK, dtype=torch.int32, device="cuda")
        for r, s in enumerate(lens):
            n = -(-s // BLK)
            bt[r, :n] = torch.randint(0, POOL_SLOTS, (n,), device="cuda", generator=gen, dtype=torch.int32)
        out.append(dict(
            bt=bt,
            iq=(torch.randn(rows, hkv, D, device="cuda", generator=gen) * 0.5).to(torch.float8_e4m3fn),
            q=torch.randn(rows, 16 * hkv, D, device="cuda", generator=gen).bfloat16(),
        ))
    return seq, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hkv", type=int, default=2)
    p.add_argument("--ql", type=int, nargs="*", default=[8])
    p.add_argument("--reqs", type=int, nargs="*", default=[1, 2, 4, 8, 16, 24, 32, 40, 48, 64])
    p.add_argument("--calls", type=int, default=50)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--select-max-rows", type=int, help="host.SELECT_MAX_ROWS[hkv] for the fused side (sweeps)")
    a = p.parse_args()
    if a.select_max_rows:
        host.SELECT_MAX_ROWS[a.hkv] = a.select_max_rows
    gen = torch.Generator(device="cuda").manual_seed(0)
    icache, kv = make_pool(a.hkv, gen)
    one = torch.ones((), dtype=torch.float32, device="cuda")
    print("| KV heads | ql | requests | rows | fused us | chain us | fused vs chain |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    with torch.inference_mode():
        for ql in a.ql:
            for reqs in a.reqs:
                rows = reqs * ql
                seq, calls = make_calls(a.hkv, ql, reqs, a.calls, gen)
                width = iref.score_width(MAX_MODEL_LEN)
                score = torch.empty(a.hkv, rows, width, dtype=torch.float32, device="cuda")
                ids = {k: torch.empty(a.hkv, rows, 16, dtype=torch.int32, device="cuda") for k in ("fused", "chain")}
                out = {k: torch.empty(rows, 16 * a.hkv, D, dtype=torch.bfloat16, device="cuda")
                       for k in ("fused", "chain")}
                quant = {k: host._quant_buffers(rows, 2048 * a.hkv, "cuda") for k in ("fused", "chain")}

                def side(name, c):
                    if name == "fused":
                        nvfp4_index_score_rows(c["iq"], icache, score, c["bt"], seq, INIT, LOCAL, query_len=ql,
                                               max_seq_len=MAX_MODEL_LEN)
                        extra = dict(index_score=score, index_max_seq_len=MAX_MODEL_LEN)
                    else:
                        nvfp4_index_score_top16(c["iq"], icache, c["bt"], seq, INIT, LOCAL, ids[name], query_len=ql,
                                                max_seq_len=MAX_MODEL_LEN, score=score)
                        extra = {}
                    host.nvfp4_sparse_decode(c["q"], kv, ids[name], c["bt"], seq, D**-0.5, out[name], k_scale=one,
                                             v_scale=one, decode_query_len=ql, quant_out=quant[name],
                                             compute_type=torch.float16, **extra)

                for name in ("fused", "chain"):
                    for t in quant[name][:2]:
                        t.zero_()
                    side(name, calls[0])
                torch.cuda.synchronize()
                assert torch.equal(ids["fused"], ids["chain"])
                assert all(torch.equal(x, y) for x, y in zip(quant["fused"][:2], quant["chain"][:2], strict=True))
                graphs = {}
                for name in ("fused", "chain"):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        for c in calls:
                            side(name, c)
                    graphs[name] = g
                times = {k: [] for k in graphs}
                for g in graphs.values():
                    g.replay()
                torch.cuda.synchronize()
                for r in range(a.rounds):
                    for name in (("fused", "chain") if r % 2 == 0 else ("chain", "fused")):
                        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                        s.record()
                        for _ in range(a.replays):
                            graphs[name].replay()
                        e.record()
                        torch.cuda.synchronize()
                        times[name].append(s.elapsed_time(e) * 1e3 / (a.replays * len(calls)))
                f, c = statistics.median(times["fused"]), statistics.median(times["chain"])
                print(f"| {a.hkv} | {ql} | {reqs} | {rows} | {f:.2f} | {c:.2f} | {100 * (f / c - 1):+.1f}% |",
                      flush=True)
                del graphs, calls
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
