# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of the norm / RoPE / NVFP4 insert kernels on gfx950.

Timing (``timeit_graph``): a HIP graph holds 100 calls, each on its own inputs (qkv or the
split-K slab, positions, the two slot mappings, q_out / index_q_out, and its own pages
of one large KV / index cache). The inputs rotate over more than 400 MB in all (above
the 256 MB MALL), so every call reads and writes HBM-cold data; this is conservative
for qkv, which serving gets warm from the QKV GEMM. Decode batches (up to 1024 tokens)
are requests of 4 tokens (the MTP / DSpark rows) in their own pages; larger batches
are 8 requests continuing 40K-token contexts. The first and last input are checked
against ``norm_rope_nvfp4_insert_d128_rot64_p128/reference.py`` before timing. Median of 5
rounds of 10 replays of every graph, us per call.

  python3 benchmarks/attention/norm_rope_nvfp4_insert_d128_rot64_p128/benchmark_norm_rope_insert.py   # the entry's kernel per size
  python3 benchmarks/attention/norm_rope_nvfp4_insert_d128_rot64_p128/benchmark_norm_rope_insert.py --reduce   # the split-K reduce kernel

Heads q16 kv1 iq1, rotary 64 (TP4).
"""

import argparse
import statistics
import time

import torch
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import (
    norm_rope_nvfp4_insert,
    splitk_reduce_norm_rope_nvfp4_insert,
)
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import reference as ref
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128 import host

D, BLK, WIDTH = ref.D, ref.BLK, ref.WIDTH
EPS, ROT = 1e-6, 64
HEADS = (16, 1, 1)
LLC_TARGET_BYTES = 400 * 1024**2  # above the 256 MB MALL
CALLS = 100  # calls per graph, each on its own inputs
VERBOSE = False
MAX_POS = 65536
TOKENS = [1, 4, 64, 172, 256, 512, 1024, 4096, 16384]
REDUCE = [(32, 3), (128, 3), (200, 2), (288, 2)]  # (tokens, split) of the QKV GEMM's deferred rows


def slot_plan(n, gen):
    """(positions, slots, pages): decode batches as requests of up to 4 tokens, one
    page each at a random offset; larger ones as 8 requests continuing 40K contexts."""
    if n <= 1024:
        run = min(4, n)
        reqs = -(-n // run)
        pos = []
        for r in range(reqs):
            cnt = min(run, n - r * run)
            start = int(torch.randint(0, MAX_POS - 8, (1,), generator=gen))
            pos.append(torch.arange(start, start + cnt))
        pos = torch.cat(pos)
        off = torch.cat([torch.arange(min(run, n - r * run)) for r in range(reqs)])
        start_in_page = torch.randint(0, BLK - run + 1, (reqs,), generator=gen)
        req = torch.arange(n) // run
        return pos, req * BLK + start_in_page[req] + off, reqs
    n_req, per = 8, n // 8
    pos, slot, base = [], [], 0
    for r in range(n_req):
        start = 40000 + 37 * r + 11 + (r * r) % 13
        cnt = per if r < n_req - 1 else n - per * (n_req - 1)
        p = torch.arange(start, start + cnt)
        first = start // BLK
        pos.append(p)
        slot.append((base + p // BLK - first) * BLK + p % BLK)
        base += (start + cnt - 1) // BLK - first + 1
    return torch.cat(pos), torch.cat(slot), base


class Inputs:
    """``copies`` input sets of one shape, each in its own pages."""

    def __init__(self, n, split_k=0, seed=0):
        self.n, (self.nq, self.nkv, self.niq) = n, HEADS
        self.row = (self.nq + 2 * self.nkv + self.niq + 1) * D
        gen = torch.Generator().manual_seed(seed)
        pos, slots, pages = slot_plan(n, gen)
        page_bytes = 2 * self.nkv * BLK * WIDTH + BLK * WIDTH
        q_bytes = n * self.nq * D * 2
        in_bytes = n * self.row * (4 * split_k if split_k else 2)
        per_input = in_bytes + q_bytes + n * self.niq * D + 24 * n + pages * page_bytes
        self.copies = max(2, -(-LLC_TARGET_BYTES // per_input))
        c = self.copies
        dev = "cuda"
        torch.manual_seed(seed)
        if split_k:
            self.slab = torch.randn(c, split_k, n, self.row, device=dev)
        else:
            self.qkv = torch.randn(c, n, self.row, device=dev).to(torch.bfloat16)
        self.w = [(torch.randn(D, device=dev) * 0.5).to(torch.bfloat16) for _ in range(4)]
        self.cs = (torch.rand(MAX_POS, ROT, device=dev) * 2 - 1).to(torch.bfloat16)
        self.pos = pos.to(dev).repeat(c, 1)
        # input i uses pages [i * pages, (i + 1) * pages) in a random order
        perm = torch.randperm(c * pages, generator=gen).view(c, pages)
        sl = perm[:, slots // BLK] * BLK + slots % BLK
        self.slots = sl.to(dev)
        self.islots = sl.flip(0).to(dev)  # another input's pages
        self.kv = torch.zeros(c * pages, 2 * self.nkv, BLK, WIDTH, dtype=torch.uint8, device=dev)
        self.idx = torch.zeros(c * pages, BLK, WIDTH, dtype=torch.uint8, device=dev)
        self.q = torch.empty(c, n, self.nq * D, dtype=torch.bfloat16, device=dev)
        self.iq = torch.empty(c, n, self.niq * D, dtype=torch.float8_e4m3fn, device=dev)
        self.ks = torch.tensor(0.37, device=dev)
        self.vs = torch.tensor(0.21, device=dev)
        self.split_k = split_k
        self.bytes = in_bytes + n * (self.nq * D * 2 + self.niq * D + 24 + ROT * 2)
        self.bytes += n * (self.nkv * 2 + 1) * WIDTH  # the NVFP4 rows written

    def call(self, i, kernel="main", op=None):
        args = (*self.w, self.cs, self.pos[i], self.slots[i], self.islots[i], self.kv, self.idx,
                self.ks, self.vs, self.q[i], self.iq[i])  # fmt: skip
        kw = dict(num_heads=self.nq, num_kv_heads=self.nkv, num_index_heads=self.niq, eps=EPS)
        if self.split_k:
            fn = op or splitk_reduce_norm_rope_nvfp4_insert
            fn(self.slab[i], self.split_k, *args, **kw)
        else:
            fn = op or norm_rope_nvfp4_insert
            fn(self.qkv[i], *args, **kw, prefill=kernel == "prefill")

    def verify(self, i, kernel):
        """Input ``i`` against the reference: q_out, index_q_out, the pages at its slots."""
        self.call(i, kernel)
        got = (self.q[i].clone(), self.iq[i].clone(), self.kv.clone(), self.idx.clone())
        self.call(i, kernel, op=getattr(ref, "splitk_reduce_norm_rope_nvfp4_insert" if self.split_k else "norm_rope_nvfp4_insert"))
        torch.cuda.synchronize()
        want = (self.q[i], self.iq[i], self.kv, self.idx)
        pairs = [(got[0].float(), want[0].float()), (got[1].float(), want[1].float())]
        kc, ks, vc, vs = ref.read_main_pages(got[2], self.slots[i])
        kc_r, ks_r, vc_r, vs_r = ref.read_main_pages(want[2], self.slots[i])
        pairs += [(ref.nvfp4_dequantize(kc, ks, self.ks), ref.nvfp4_dequantize(kc_r, ks_r, self.ks)),
                  (ref.nvfp4_dequantize(vc, vs, self.vs), ref.nvfp4_dequantize(vc_r, vs_r, self.vs))]  # fmt: skip
        for a, b in pairs:
            a, b = a.double().flatten(), b.double().flatten()
            cos = (a @ b).item() / max(a.norm().item() * b.norm().item(), 1e-300)
            assert cos >= ref.COS_MIN, (self.n, kernel, i, cos)


def capture(inputs, kernel="main"):
    graphs = []
    for g0 in range(0, inputs.copies, CALLS):
        ids = [(g0 + j) % inputs.copies for j in range(CALLS)]
        for i in ids[:2]:
            inputs.call(i, kernel)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in ids:
                inputs.call(i, kernel)
        graphs.append(graph)
    torch.cuda.synchronize()
    return graphs


QUEUE = 32  # graph launches in flight at most


def replay_ms(launches):
    """GPU time (ms) of the graph launches, at most QUEUE in flight: chunks of QUEUE
    launches, each between two events and followed by a synchronize, their times
    summed (the host gaps between chunks are not counted). With hundreds of graph
    launches queued at once the HIP runtime stalled now and then (the
    host spinning in sched_yield, the GPU idle)."""
    total = 0.0
    for c0 in range(0, len(launches), QUEUE):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for g in launches[c0 : c0 + QUEUE]:
            g.replay()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total


def time_graphs(graphs, replays=10, rounds=5, warmup=3):
    """Median over rounds of the mean us per call."""
    replay_ms(graphs * warmup)
    calls = replays * len(graphs) * CALLS
    return statistics.median(replay_ms(graphs * replays) / calls for _ in range(rounds)) * 1e3


def log(msg):
    if VERBOSE:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def bench(n, kernels, split_k=0, rounds=1):
    inputs = Inputs(n, split_k)
    log(f"n={n}: {inputs.copies} inputs")
    for kernel in kernels:
        inputs.verify(0, kernel)
        inputs.verify(inputs.copies - 1, kernel)
    log("verified")
    graphs = {k: capture(inputs, k) for k in kernels}
    log(f"captured {sum(len(g) for g in graphs.values())} graphs")
    times = {k: [] for k in kernels}
    for _ in range(rounds):
        for k in kernels:
            times[k].append(time_graphs(graphs[k]))
    for kernel, ts in times.items():
        name = "splitk" if split_k else kernel
        us = " / ".join(f"{t:.2f}" for t in ts)
        what = f"split {split_k}, " if split_k else ""
        if split_k:
            tp = host._reduce_token_pairs(n, split_k)
        elif n >= host.PREFILL_TOKENS:
            name, tp = "prefill", host._prefill_token_pairs(n)
        else:
            tp = host._main_token_pairs(n)
        print(f"q16_kv1_iq1 n={n} {name}: {what}pairs {tp}, {inputs.copies} inputs; "
              f"{us} us ({inputs.bytes / statistics.median(ts) / 1e3:.0f} GB/s)", flush=True)  # fmt: skip
    del graphs, inputs
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="*", default=TOKENS)
    parser.add_argument("--reduce", action="store_true", help="the split-K reduce kernel")
    parser.add_argument("--llc-mb", type=int, default=400, help="footprint the inputs rotate over")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    global LLC_TARGET_BYTES, VERBOSE
    LLC_TARGET_BYTES, VERBOSE = args.llc_mb * 1024**2, args.verbose
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("requires gfx950")
    with torch.inference_mode():
        if args.reduce:
            for n, split_k in REDUCE:
                bench(n, ["main"], split_k)
        else:
            for n in args.tokens:
                bench(n, ["main"])


if __name__ == "__main__":
    main()
