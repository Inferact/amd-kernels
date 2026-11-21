# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the ``mla_decode_common`` entries add over a direct shape-package call, on gfx950.

Graph replay: one graph of ``--copies`` entry calls and one of the same direct calls (same operands), replayed
alternately, min of 2 x ``--rounds`` replays, us per call. Host: the eager dispatch with the routed package function
replaced by a no-op, minus the no-op called directly (the Python the entry adds per eager call), and one
``<op>_unsupported_reason`` call; min of ``--rounds`` loops of 2000 calls. ``--ranks8``: the two 8-rank ops' graph
replay on 8 GPUs (max over ranks).
"""

import argparse
import os
import socket
import time

import torch

from amd_kernels import warm_up
from amd_kernels.attention import mla_decode_common as E
from amd_kernels.attention.mla_decode_common import combine, dense, prologue, query, sparse

BF16, FP8 = torch.bfloat16, torch.float8_e4m3fn


def graph_pair_us(direct, entry, copies, rounds):
    """(direct, entry) us per call: one graph of ``copies`` calls each, replayed in alternating order, min of rounds."""
    graphs = []
    for call in (direct, entry):
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(copies):
                call()
        graphs.append(graph)
    times = ([], [])
    for i in range(2 * rounds):
        for j in ((0, 1) if i % 2 else (1, 0)):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[j].replay()
            end.record()
            torch.cuda.synchronize()
            times[j].append(start.elapsed_time(end) * 1e3 / copies)
    return min(times[0]), min(times[1])


def host_us(call, rounds, n=2000):
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(n):
            call()
        best = min(best, (time.perf_counter() - t0) * 1e6 / n)
    return best


def z(*shape, dtype=BF16):
    return torch.zeros(shape, dtype=dtype, device="cuda")


def r(*shape, dtype=BF16, scale=1.0):
    return (torch.randn(shape, device="cuda") * scale).to(dtype)


def insert_args(m):
    side = dict(ln_weight=r(128), ln_bias=r(128), cache=z(64, 64, 68, dtype=torch.uint8),
                slot_mapping=torch.arange(m, device="cuda"), weights=z(m, 32, dtype=torch.float32), weight_scale=0.1)
    return (r(m, 2272), r(1536), r(512), 1e-6, z(m, 1536), z(m, 512), z(64, 64, 656, dtype=torch.uint8),
            torch.arange(m, device="cuda") + 64, side)


def prep_args(m):
    return (r(m, 32, 128), None, None, None, None, 1e-6, None, z(64, 64, 68, dtype=torch.uint8),
            z(m, 2048, dtype=torch.int32), 0.1, z(m, 32, 64, dtype=torch.uint8), z(m, 1, 4, 16, 4, dtype=torch.uint8),
            None)


def sparse_args(m, keys=256):
    cache = z(16384, 656, dtype=torch.uint8)
    cache[:, :512] = r(16384, 512, scale=0.5, dtype=FP8).view(torch.uint8)
    cache[:, 512:528] = torch.ones(16384, 4, device="cuda").view(torch.uint8)
    slots = torch.randint(0, 16384, (m * keys,), device="cuda", dtype=torch.int32)
    indptr = torch.arange(0, (m + 1) * keys, keys, device="cuda", dtype=torch.int32)
    return (r(m, 96, 576, scale=0.5), cache, indptr, slots, 576**-0.5)


def dense_args(qlen, reqs, local=2000):
    idx = torch.randperm(reqs * local + 64, device="cuda")[: reqs * local].int()
    ip = torch.arange(0, (reqs + 1) * local, local, dtype=torch.int32, device="cuda")
    one = torch.ones(1, device="cuda")
    return (r(reqs * qlen, 96, 576, dtype=FP8), r(reqs * local + 64, 576, dtype=FP8), ip, idx, ip * 8, one, one,
            192**-0.5, qlen, 3, 8)


def single(args):
    pkg_insert = prologue._shape_host((1536, 512))
    pkg_prep = prologue._index_host((32, 128))
    pkg_sparse = sparse._shape_host((96, 576, 656))
    pkg_dense = dense._shape_host((96, 576))
    cases = []
    for m in (12, 192):
        a = insert_args(m)
        direct = (*a[:8], pkg_insert.IndexSide(**a[8]))
        cases.append((f"qkv_a_norm_kv_insert + index, M {m}", lambda a=a: E.qkv_a_norm_kv_insert(*a),
                      lambda d=direct: pkg_insert.qkv_a_norm_kv_insert(*d), pkg_insert, "qkv_a_norm_kv_insert", a,
                      E.qkv_a_norm_kv_insert, E.qkv_a_norm_kv_insert_unsupported_reason))
    for m in (12, 192):
        a = prep_args(m)
        cases.append((f"indexer_prep, M {m}", lambda a=a: E.indexer_prep(*a), lambda a=a: pkg_prep.indexer_prep(*a),
                      pkg_prep, "indexer_prep", a, E.indexer_prep, E.indexer_prep_unsupported_reason))
    for m in (12, 192):
        a = sparse_args(m)
        cases.append((f"fp8_sparse_mla_decode, M {m}", lambda a=a: E.fp8_sparse_mla_decode(*a),
                      lambda a=a: pkg_sparse.fp8_sparse_mla_decode(*a), pkg_sparse, "fp8_sparse_mla_decode", a,
                      E.fp8_sparse_mla_decode, E.fp8_sparse_mla_decode_unsupported_reason))
    for qlen, reqs in ((4, 13), (3, 48)):
        a = dense_args(qlen, reqs)
        cases.append((f"fp8_dense_mla_decode, {reqs} x {qlen} rows", lambda a=a: E.fp8_dense_mla_decode(*a),
                      lambda a=a: pkg_dense.fp8_dense_mla_decode(*a), pkg_dense, "fp8_dense_mla_decode", a,
                      E.fp8_dense_mla_decode, E.fp8_dense_mla_decode_unsupported_reason))
    print("| op, shape | direct us | entry us | added us | host: dispatch added us | host: reason us |")
    print("|---|---|---|---|---|---|")
    for name, entry, direct, pkg, attr, a, op, reason in cases:
        assert reason(*a) is None, (name, reason(*a))
        g_direct, g_entry = graph_pair_us(direct, entry, args.copies, args.rounds)
        real = getattr(pkg, attr)
        noop = lambda *x, **k: None  # noqa: E731
        setattr(pkg, attr, noop)
        try:
            added = host_us(lambda: op(*a), args.rounds) - host_us(lambda: noop(*a), args.rounds)
        finally:
            setattr(pkg, attr, real)
        why = host_us(lambda: reason(*a), args.rounds)
        print(f"| {name} | {g_direct:.2f} | {g_entry:.2f} | {g_entry - g_direct:+.2f} | {added:.2f} | {why:.2f} |")
    host_comm(args)


def host_comm(args):
    """The comm-routed ops' added host time and reason, on communicator objects that launch nothing."""
    for mod, cls, attr, op, reason, a in (
        (query, "QueryGatherComm", "q_b_allgather_wuk_all_heads", E.q_bproj_wuk_gather,
         E.q_bproj_wuk_gather_unsupported_reason, lambda c, m: (c, z(m, 1536), z(2304, 1536), z(96, 8, 4, 4, 4, 16, 8),
                                                                 z(m, 96, 576))),
        (combine, "DcpVupCombine", "dcp8_lse_combine_vup", E.lse_combine_vup, E.lse_combine_vup_unsupported_reason,
         lambda c, m: (c, z(m, 96, 512), z(m, 96, dtype=torch.float32), z(96, 128, 512), z(m, 12, 128))),
    ):
        pkg = mod._shape_host(next(iter(mod._SHAPE_HOSTS)))
        comm = object.__new__(getattr(pkg, cls))
        comm.mode, comm.max_tokens = "q", 192
        operands = a(comm, 12)
        real = getattr(pkg, attr)
        noop = lambda *x, **k: None  # noqa: E731
        setattr(pkg, attr, noop)
        try:
            added = host_us(lambda: op(*operands), args.rounds) - host_us(lambda: noop(*operands), args.rounds)
        finally:
            setattr(pkg, attr, real)
        why = host_us(lambda: reason(*operands), args.rounds)
        print(f"| {op.__name__} (8 ranks), host only | | | | {added:.2f} | {why:.2f} |")


def _rank_main(rank, port, args, results):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=8)
    group, dev = dist.group.WORLD, torch.device("cuda", rank)
    warm_up.set_workers(4)
    q_pkg = query._shape_host((1536, 2304, 96, 8))
    c_pkg = combine._shape_host((96, 128, 512, 8))
    qc = E.q_gather_comm(rank, group, dev, "q", q_lora=1536, q_b_rows=2304, heads=96)
    cc = E.lse_combine_comm(rank, group, dev, heads=96, v_head=128, kv_lora=512)
    wq, wu = E.pack_q_b(r(2304, 1536, scale=0.03)), E.pack_w_uk(r(96, 128, 512, scale=0.05))
    wt = E.prepare_w_uv(r(96, 512, 128, scale=0.05))
    rows = []
    for m in (12, 192):
        q_a, out = r(m, 1536), z(m, 96, 576)
        x, lse, o = r(m, 96, 512, scale=0.5), r(m, 96, dtype=torch.float32), z(m, 12, 128)
        for name, entry, direct in (
            (f"q_bproj_wuk_gather, M {m}", lambda: E.q_bproj_wuk_gather(qc, q_a, wq, wu, out=out),
             lambda: q_pkg.q_b_allgather_wuk_all_heads(qc, q_a, wq, wu, out=out)),
            (f"lse_combine_vup, M {m}", lambda: E.lse_combine_vup(cc, x, lse, wt, o),
             lambda: c_pkg.dcp8_lse_combine_vup(cc, x, lse, wt, o)),
        ):
            rows.append((name, *graph_pair_us(direct, entry, args.copies, args.rounds)))
    torch.cuda.synchronize()
    dist.barrier()
    qc.close()
    results.put((rank, rows))
    dist.destroy_process_group()


def ranks8(args):
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    procs = [ctx.Process(target=_rank_main, args=(rk, port, args, results)) for rk in range(8)]
    for p in procs:
        p.start()
    per_rank = dict(results.get(timeout=600) for _ in range(8))
    for p in procs:
        p.join()
    print("| op, shape (8 ranks, max over ranks) | direct us | entry us | added us |")
    print("|---|---|---|---|")
    for i, (name, _, _) in enumerate(per_rank[0]):
        d = max(per_rank[k][i][1] for k in per_rank)
        e = max(per_rank[k][i][2] for k in per_rank)
        print(f"| {name} | {d:.2f} | {e:.2f} | {e - d:+.2f} |")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--copies", type=int, default=20)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--ranks8", action="store_true")
    args = p.parse_args()
    warm_up.set_workers(int(os.environ.get("AKR_WARM_WORKERS", "8")))
    if args.ranks8:
        ranks8(args)
    else:
        single(args)


if __name__ == "__main__":
    main()
