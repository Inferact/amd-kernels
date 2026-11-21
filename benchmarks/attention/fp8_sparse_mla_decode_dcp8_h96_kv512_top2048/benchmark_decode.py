# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``fp8_sparse_mla_decode`` against AITER's Triton ``sparse_mla_fwd`` (vLLM's decode path at
96 heads) and the roofline, on gfx950, inputs cold.

One HIP graph holds ``--sets`` launches, each with its own q, output, CSR slots (scattered over the pages of a
``--cache-gb`` cache, a request's rows sharing ~85% of their keys) and indptr; the sets total more than the 256 MB MALL.
Median of 20 replays in us per launch. Roofline: bytes = q read + output and LSE write + unique KV records (656 B) + slots,
at 80% of 8 TB/s; FLOPs = 2 * rows * 96 * keys * (576 + 512) at a bf16 MFMA rate of 1.25 PF (the q.k part
runs in fp8 here, so it is conservative). The first set of each size is checked against ``reference.py``.
"""

import argparse
import math
import statistics

import torch
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import fp8_sparse_mla_decode
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import (
    D,
    H,
    RECORD_BYTES,
    check_close,
    reference,
)
from tests.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.cases import PAGE, SERVED_ROWS, SM_SCALE, fill_slots

HBM = 8.0e12 * 0.8
BF16_PEAK = 1.25e15
SET_BYTES = 600e6


def stock(q, cache, indptr, slots, out):
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

    return sparse_mla_fwd(
        q, cache.unsqueeze(1), indptr, slots, SM_SCALE, kv_lora_rank=512, qk_rope_head_dim=64, has_invalid=False,
        dot_precision="bf16", out=out, return_lse=True,
    )


def make_set(rows, mean_keys, spec, ctx_local, pages, gen):
    nseq = math.ceil(rows / spec)
    ptab = torch.randperm(pages, device="cuda", generator=gen)[: nseq * (ctx_local // PAGE + 1)].view(nseq, -1)
    counts, picks = [], []
    for s in range(nseq):
        base = torch.randperm(ctx_local, device="cuda", generator=gen)[: int(mean_keys * 1.15)]
        for _ in range(spec):
            if len(counts) == rows:
                break
            n = min(int(torch.distributions.Binomial(2048, torch.tensor(mean_keys / 2048)).sample()), base.numel())
            keep = base[torch.randperm(base.numel(), device="cuda", generator=gen)[:n]]
            picks.append((ptab[s][keep // PAGE] * PAGE + keep % PAGE).int())
            counts.append(n)
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    indptr[1:] = torch.tensor(counts, device="cuda").cumsum(0).int()
    q = (torch.randn(rows, H, D, device="cuda", generator=gen) * 0.5).bfloat16()
    return q, indptr, torch.cat(picks), picks


def time_graph(fns, replays=20):
    for f in fns:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in fns:
            f()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(fns))
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=",".join(map(str, SERVED_ROWS)))
    p.add_argument("--keys", type=int, default=256, help="mean keys per row on this rank (256 = top-2048 over 8 ranks)")
    p.add_argument("--spec", type=int, default=4)
    p.add_argument("--ctx-local", type=int, default=19000)
    p.add_argument("--cache-gb", type=float, default=4.0)
    p.add_argument("--max-sets", type=int, default=192)
    p.add_argument("--skip-stock", action="store_true")
    args = p.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    pages = int(args.cache_gb * 2**30) // RECORD_BYTES // PAGE
    cache = torch.empty(pages * PAGE, RECORD_BYTES, dtype=torch.uint8, device="cuda")
    cpu_gen = torch.Generator().manual_seed(1)
    chunk = torch.empty(1 << 16, RECORD_BYTES, dtype=torch.uint8)
    chunk_cuda = torch.empty_like(chunk, device="cuda")
    fill_slots(chunk_cuda, torch.arange(chunk.shape[0]), cpu_gen)
    for i in range(0, cache.shape[0], chunk.shape[0]):
        n = min(chunk.shape[0], cache.shape[0] - i)
        cache[i : i + n] = chunk_cuda[:n]
    print(f"# DCP8 sparse MLA decode, 96 heads, ~{args.keys} keys/row, {args.spec} rows/request, cache {cache.numel() / 2**30:.1f} GiB")
    print("rows | stock us | new us | speedup | bytes MB | GFLOP | roofline us | new % roofline | new TB/s | new TFLOP/s | stock % roofline")
    for rows in map(int, args.rows.split(",")):
        sets = max(2, min(args.max_sets, int(SET_BYTES // (rows * H * (D + 512) * 2 + 1e6))))
        data = [make_set(rows, args.keys, args.spec, args.ctx_local, pages, gen) for _ in range(sets)]
        outs = [torch.empty(rows, H, 512, dtype=torch.bfloat16, device="cuda") for _ in range(sets)]
        o, lse = fp8_sparse_mla_decode(data[0][0], cache, data[0][1], data[0][2], SM_SCALE)
        ref_o, ref_lse = reference(data[0][0], cache, data[0][1], data[0][2], SM_SCALE)
        check_close(o.cpu(), lse.cpu(), ref_o.cpu(), ref_lse.cpu())
        new = time_graph([lambda d=d, ou=ou: fp8_sparse_mla_decode(d[0], cache, d[1], d[2], SM_SCALE, ou) for d, ou in zip(data, outs)])
        old = 1e9 if args.skip_stock else time_graph([lambda d=d, ou=ou: stock(d[0], cache, d[1], d[2], ou) for d, ou in zip(data, outs)])
        keys = data[0][2].numel() / rows
        uniq = statistics.mean(torch.unique(d[2]).numel() for d in data[:4])
        nbytes = rows * H * D * 2 + rows * H * 512 * 2 + rows * H * 4 + uniq * RECORD_BYTES + keys * rows * 4
        flop = 2 * rows * H * keys * (D + 512)
        roof = max(nbytes / HBM, flop / BF16_PEAK) * 1e6
        print(
            f"{rows} | {old:.1f} | {new:.1f} | {old / new:.2f}x | {nbytes / 1e6:.1f} | {flop / 1e9:.2f} | {roof:.1f} | "
            f"{roof / new * 100:.0f}% | {nbytes / new / 1e6:.2f} | {flop / new / 1e6:.0f} | {roof / old * 100:.0f}%"
        )
        del data, outs


if __name__ == "__main__":
    main()
