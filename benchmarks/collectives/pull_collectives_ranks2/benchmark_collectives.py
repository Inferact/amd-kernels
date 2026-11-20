# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The SP-prefill collectives on two ranks vs RCCL at world 2, per served token count T (shard rows R = T / 2).

Graph replay: ``CALLS`` distinct inputs per graph (rotating beyond the 256 MB MALL), 10 replays per round, 5 rounds in
alternating order, the max over ranks of each round, median over rounds; us per call. The unpacking all-to-all keeps
host sequence state (eager only), so it and its baseline are timed eagerly (back-to-back calls, same rounds).

Entries (left) and their baseline (right), the same bytes on the link:
- ``moe_tail``: ``slots4_reduce_scatter_add_rmsnorm_mxfp8`` vs slot reduce + shared (one launch) + RCCL reduce-scatter
  + the row-major norm; ``x3`` / ``bf16_link`` the one-launch tail with the coded / BF16 link; ``two_launch`` the same
  tail as stage + pull and a norm launch; ``rccl_ar`` the non-SP path's RCCL all-reduce of the same [T, H] bf16.
  Roofline: the default communicator's link bytes (9600 B per coded row, 12288 bf16). ``moe_tail_owner``: the same
  tail with the owner-form shared term ([R, H], the own rows only) against its bf16-link form.
- ``rs``: ``reduce_scatter`` vs RCCL reduce-scatter of [T, H] bf16.
- ``compact_ag``: ``all_gather`` of the six compact MoE fields (7328 B/row) vs RCCL all-gather of the same bytes.
- ``routed_ag``: ``all_gather`` of the four routed MoE fields (3296 B/row) vs RCCL all-gather of the same bytes.
- ``codes_ag``: ``all_gather`` of MXFP8 codes + scales (6336 B/row) vs RCCL all-gather of the same bytes.
- ``norm_ag``: ``norm_all_gather`` vs the norm + RCCL all-gather of codes + scales.
- ``a2a``: ``all_to_all_unpack_shuffled`` of rows the producer wrote into the pool slot (4224 B/row each way) vs RCCL
  all-to-all + torch unpack + the GEMM's scale shuffle (eager).

Roofline: per-rank bytes over the link / ``LINK_GBS`` (the best measured rate of one link with both directions busy:
SDMA, ``bench_engines.py``); ``CU_PULL_GBS`` is the CU pull ceiling (``benchmark_link.py``).

  python3 benchmarks/collectives/pull_collectives_ranks2/benchmark_collectives.py [out.json] [entry ...] \
      [--tokens T ...]
"""
import json
import statistics
import sys

import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

H = 6144
D = 4096
ROUNDS = 5
REPLAYS = 10
EPS = 1e-6
LINK_GBS = 61.2
CU_PULL_GBS = 57.4
MALL = 256 << 20
ENTRIES = ("moe_tail", "moe_tail_owner", "rs", "routed_ag", "compact_ag", "codes_ag", "norm_ag", "a2a")
COMPACT = (192, 16, 16, 6144, 192, 768)
ROUTED = (3072, 192, 16, 16)


def _calls(bytes_per_call):
    return max(2, min(16, -(-2 * MALL // bytes_per_call)))


def _time_graph(g, calls):
    dist.barrier()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(REPLAYS):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (REPLAYS * calls)


def _time_eager(fn, calls):
    fn()
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(2):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (2 * calls)


def _graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    return g


def _measure(arms, calls, eager=False):
    """arms: name -> fn over all calls; -> name -> rank-max per round, median over rounds (us per call)."""
    run = {k: (fn if eager else _graph(fn)) for k, fn in arms.items()}
    times = {k: [] for k in arms}
    for r in range(ROUNDS):
        keys = list(arms) if r % 2 == 0 else list(reversed(arms))
        for k in keys:
            t = _time_eager(run[k], calls) if eager else _time_graph(run[k], calls)
            every = [None, None]
            dist.all_gather_object(every, t)
            times[k].append(max(every))
    return {k: statistics.median(v) for k, v in times.items()}


def _bf16(shape, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randn(shape, generator=g, device=dev, dtype=torch.float32).bfloat16()


def _u8(shape, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    return torch.randint(0, 120, shape, generator=g, device=dev, dtype=torch.uint8)


def _worker(rank, errors, entries, tokens_list, out):
    from amd_kernels.collectives.pull_collectives_ranks2 import (
        AllGatherComm, Collectives, FencedAllGatherComm, NormAllGatherComm, Pools, SlotsReduceScatterComm)
    from amd_kernels.gemm.mxfp8_common.gemm import _preshuffle_activation_scale
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.common.reduce_slots_shared import reduce_slots_shared
    from amd_kernels.norm.gemma_add_rmsnorm_h6144 import add_rmsnorm as an

    dev = torch.device("cuda", rank)
    nccl = dist.new_group(backend="nccl")
    max_t = max(tokens_list)
    w = (_bf16((H,), 3, dev) * 0.2).bfloat16()
    results = {}
    comms = {}
    if {"moe_tail", "moe_tail_owner", "rs"} & set(entries):
        comms["rs"] = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, max_rows=max_t)
        comms["rs"].warm_up()
    if {"moe_tail", "moe_tail_owner"} & set(entries):
        for coded in (True, False):
            comms[coded] = SlotsReduceScatterComm(rank, dist.group.WORLD, dev, max_rows=max_t, coded=coded)
            comms[coded].warm_up()
    if {"compact_ag", "routed_ag", "codes_ag"} & set(entries):
        comms["ag"] = AllGatherComm(rank, dist.group.WORLD, dev, (max_t // 2) * 7328 + 6 * 256)
        comms["ag"].warm_up()
    if "norm_ag" in entries:
        comms["norm"] = NormAllGatherComm(rank, dist.group.WORLD, dev, max_rows=max_t // 2)
        comms["norm"].warm_up()
    if "a2a" in entries:
        pools = Pools(rank, 2, dist.group.WORLD, dev, max_t * (D + D // 32) + 4096, sets=1)
        comms["a2a"] = FencedAllGatherComm(pools)
        comms["a2a"].warm_up()
    for T in tokens_list:
        R = T // 2
        for entry in entries:
            arms, eager = {}, False
            if entry == "moe_tail":
                calls = _calls(T * 5 * H * 2)
                slots = [_bf16((T, 4, H), 100 * i + rank, dev) for i in range(calls)]
                shared = [_bf16((T, H), 100 * i + 50 + rank, dev) for i in range(calls)]
                res = [_bf16((R, H), 100 * i + 70 + rank, dev) for i in range(calls)]
                rs_out = torch.empty((R, H), dtype=torch.bfloat16, device=dev)
                ar_buf = [torch.empty((T, H), dtype=torch.bfloat16, device=dev) for _ in range(calls)]

                def fused():
                    for s, h, r in zip(slots, shared, res):
                        comms["rs"].slots4_reduce_scatter_add_rmsnorm_mxfp8(s, h, r, w, EPS)

                def baseline():
                    for s, h, r in zip(slots, shared, res):
                        x = reduce_slots_shared(s, h)
                        dist.reduce_scatter_tensor(rs_out, x, group=nccl)
                        an.gemma_add_rmsnorm_mxfp8_row_major(rs_out, r, w, EPS)

                def two_launch():
                    for s, h, r in zip(slots, shared, res):
                        comms["rs"].slots4_reduce_scatter_add_rmsnorm_mxfp8(s, h, r, w, EPS, fused=False)

                def rccl_ar():
                    for x in ar_buf:
                        dist.all_reduce(x, group=nccl)

                def link_form(coded):
                    def run():
                        for s, h, r in zip(slots, shared, res):
                            comms[coded].slots4_reduce_scatter_add_rmsnorm_mxfp8(s, h, r, w, EPS)
                    return run
                arms = {"ours": fused, "baseline": baseline, "x3": link_form(True), "bf16_link": link_form(False),
                        "two_launch": two_launch, "rccl_ar": rccl_ar}
                link = R * (9600 if comms["rs"].coded else H * 2)
            elif entry == "moe_tail_owner":
                calls = _calls(T * 4 * H * 2 + R * H * 2)
                slots = [_bf16((T, 4, H), 100 * i + rank, dev) for i in range(calls)]
                shared = [_bf16((R, H), 100 * i + 50 + rank, dev) for i in range(calls)]
                res = [_bf16((R, H), 100 * i + 70 + rank, dev) for i in range(calls)]

                def owner_form(coded):
                    def run():
                        for s, h, r in zip(slots, shared, res):
                            comms[coded].slots4_reduce_scatter_add_rmsnorm_mxfp8(s, h, r, w, EPS)
                    return run
                arms = {"ours": owner_form(True), "baseline": owner_form(False)}
                link = R * 9600
            elif entry == "rs":
                calls = _calls(T * H * 2)
                xs = [_bf16((T, H), 100 * i + rank, dev) for i in range(calls)]
                outs = torch.empty((R, H), dtype=torch.bfloat16, device=dev)

                def ours():
                    for x in xs:
                        comms["rs"].reduce_scatter(x, outs)

                def baseline():
                    for x in xs:
                        dist.reduce_scatter_tensor(outs, x, group=nccl)
                arms = {"ours": ours, "baseline": baseline}
                link = R * H * 2
            elif entry in ("compact_ag", "routed_ag", "codes_ag"):
                cols = {"compact_ag": COMPACT, "routed_ag": ROUTED, "codes_ag": (H, H // 32)}[entry]
                row_bytes = sum(cols)
                calls = _calls(R * row_bytes * 3)
                ins = [[_u8((R, c), 100 * i + j + rank, dev) for j, c in enumerate(cols)] for i in range(calls)]
                outs = [[torch.empty((T, c), dtype=torch.uint8, device=dev) for c in cols] for _ in range(calls)]
                flat_in = [_u8((R * row_bytes,), 100 * i + rank, dev) for i in range(calls)]
                flat_out = [torch.empty((T * row_bytes,), dtype=torch.uint8, device=dev) for _ in range(calls)]

                def ours():
                    for i, o in zip(ins, outs):
                        comms["ag"].all_gather(o, i)

                def baseline():
                    for i, o in zip(flat_in, flat_out):
                        dist.all_gather_into_tensor(o, i, group=nccl)
                arms = {"ours": ours, "baseline": baseline}
                link = R * row_bytes
            elif entry == "norm_ag":
                calls = _calls(R * H * 2 * 4)
                xs = [_bf16((R, H), 100 * i + rank, dev) for i in range(calls)]
                rs_ = [_bf16((R, H), 100 * i + 50 + rank, dev) for i in range(calls)]
                q = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=dev)
                sc = torch.empty((T, H // 32), dtype=torch.uint8, device=dev)
                flat = torch.empty((T * (H + H // 32),), dtype=torch.uint8, device=dev)

                def ours():
                    for x, r in zip(xs, rs_):
                        comms["norm"].norm_all_gather(x, r, w, EPS, q_out=q, scale_out=sc)

                def baseline():
                    for x, r in zip(xs, rs_):
                        cq, cs, _ = an.gemma_add_rmsnorm_mxfp8_row_major(x, r, w, EPS)
                        local = torch.cat((cq.view(torch.uint8).reshape(-1), cs[:R].reshape(-1)))
                        dist.all_gather_into_tensor(flat, local, group=nccl)
                arms = {"ours": ours, "baseline": baseline}
                link = R * (H + H // 32)
            elif entry == "a2a":
                eager = True
                calls = _calls(T * (D + D // 32) * 2)
                packed = [_u8((T, D + D // 32), 100 * i + rank, dev) for i in range(calls)]
                recv = torch.empty((T, D + D // 32), dtype=torch.uint8, device=dev)
                a2a = comms["a2a"]

                def ours():
                    # the producer (the pack) writes into the slot: no staging copy
                    for _ in packed:
                        c = Collectives(a2a.p, 0).bind(torch.cuda.current_stream())
                        a2a.all_to_all_unpack_shuffled(c, c.alloc((T, D + D // 32), torch.uint8))

                def baseline():
                    for x in packed:
                        dist.all_to_all_single(recv, x, group=nccl)
                        blocks = recv.view(2, R, D + D // 32).transpose(0, 1)
                        codes = blocks[:, :, :D].reshape(R, 2 * D)
                        _preshuffle_activation_scale(blocks[:, :, D:].reshape(R, 2 * D // 32))
                        codes.contiguous()
                arms = {"ours": ours, "baseline": baseline}
                link = R * (D + D // 32)
            got = _measure(arms, calls, eager)
            roof = link / LINK_GBS / 1e3
            row = {"tokens": T, "link_MB": link / 1e6, "roofline_us": roof, "cu_pull_us": link / CU_PULL_GBS / 1e3,
                   **got}
            results.setdefault(entry, []).append(row)
            if rank == 0:
                extra = " ".join(f"{k} {v:.1f}" for k, v in got.items() if k not in ("ours", "baseline"))
                print(f"{entry:10s} T={T:5d} ours {got['ours']:8.1f} us  baseline {got['baseline']:8.1f} us  "
                      f"roofline {roof:7.1f} us ({roof / got['ours'] * 100:5.1f}%)  {extra}", flush=True)
            del arms
            torch.cuda.empty_cache()
    if rank == 0 and out:
        with open(out, "w") as f:
            json.dump(results, f, indent=1)


def main():
    args = sys.argv[1:]
    out = args.pop(0) if args and args[0].endswith(".json") else None
    tokens = [4096, 8192, 16384, 32768]
    if "--tokens" in args:
        i = args.index("--tokens")
        tokens = [int(a) for a in args[i + 1:]]
        args = args[:i]
    entries = tuple(args) or ENTRIES
    run_on_2_gpus(_worker, entries, tokens, out, timeout=7200)


if __name__ == "__main__":
    main()
