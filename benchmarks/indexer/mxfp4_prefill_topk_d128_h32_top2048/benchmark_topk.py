# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of ``topk_rows`` vs AITER's ``top_k_per_row_prefill`` over logits tiles larger than the LLC; roofline table.

    python3 benchmarks/indexer/mxfp4_prefill_topk_d128_h32_top2048/benchmark_topk.py [--cases 0,1,..] [json]

The roofline is the logits window read once (4 bytes per key of every row's window) at the measured HBM read ceilings
(6.5 TB/s plain loads, 6.9 TB/s nontemporal) and against a read-only kernel streaming the same window
(`read_floor`: the ceiling of this access pattern, one workgroup per row, the same loads, no compare).
"""

import json
import statistics
import sys

import torch

from amd_kernels.indexer.mxfp4_prefill_topk_d128_h32_top2048 import host
from tests.indexer.mxfp4_prefill_topk_d128_h32_top2048.inputs import (
    make_logits,
    make_windows,
)

HBM_PLAIN_TBS = 6.5
HBM_NT_TBS = 6.9
LLC_BYTES = 256 * 2**20
# (rows, width, windows): per-rank rows of a 5720-token chunk at 150K / 500K / 790K, a 4096-token chunk, a 16384-token chunk,
# a chunk of several requests, and a 2048-token chunk
CASES = [
    (512, 150_000, "causal"),
    (715, 150_000, "causal"),
    (2048, 150_000, "causal"),
    (512, 500_000, "causal"),
    (715, 500_000, "causal"),
    (715, 790_000, "causal"),
    (715, 500_000, "requests"),
    (256, 150_000, "causal"),
]


def replay_us(fn, iters=10, rounds=15):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(rounds):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / iters)
    return min(times), statistics.median(times)


def read_floor_fn(logits, ks, ke):
    """One workgroup per row reading its window with 16-byte nontemporal loads, XOR-reducing so the loads are kept."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import rocdl
    from flydsl.expr.primitive import range_constexpr
    from flydsl.expr.typing import T

    from amd_kernels.common import buffer_ops
    from amd_kernels.common.tensor_shim import _run_compiled

    I, I64 = fx.Int32, fx.Int64
    nt, u = 512, 8

    @flyc.kernel(known_block_size=[nt, 1, 1])
    def kern(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_out: fx.Int64, i64_stride: fx.Int64):
        tx = fx.thread_idx.x
        row = I(fx.block_idx.x)
        ks = I(rocdl.readfirstlane(T.i32, I(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_ks), row, vec_width=1, dtype=I))))
        ke = I(rocdl.readfirstlane(T.i32, I(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_ke), row, vec_width=1, dtype=I))))
        rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)
        base = ks & I(-64)
        nit = (ke - base + (nt * 4 * u - 1)) // (nt * 4 * u)
        acc = I(0)
        for i, st in range(0, nit, 1, init=[acc, I(0)]):
            x = I(st[0])
            for v in range_constexpr(u):
                w = fx.Vector(buffer_ops.buffer_load(rsrc, base + I(i) * (nt * 4 * u) + (v * nt + tx) * 4, vec_width=4, dtype=I, cache_modifier=2))
                x = x ^ w[0] ^ w[1] ^ w[2] ^ w[3]
            res = yield [x, I(0)]
        buffer_ops.buffer_store(res[0], buffer_ops.create_buffer_resource_from_addr(arg_out), row * nt + tx)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_out: fx.Int64, i64_stride: fx.Int64, rows: fx.Int32, stream: fx.Stream):
        kern(arg_lg, arg_ks, arg_ke, arg_out, i64_stride).launch(grid=(I64(rows), 1, 1), block=(nt, 1, 1), stream=stream)

    rows = logits.shape[0]
    sink = torch.empty(rows * nt, dtype=torch.int32, device=logits.device)

    def run():
        _run_compiled(launch, logits.data_ptr(), ks.data_ptr(), ke.data_ptr(), sink.data_ptr(), logits.stride(0), rows, torch.cuda.current_stream())

    return run


def aiter_fn(logits, ks, ke, out, stable):
    from aiter.ops.topk import top_k_per_row_prefill

    rows = logits.shape[0]

    def run():
        top_k_per_row_prefill(
            logits=logits, rowStarts=ks, rowEnds=ke, indices=out, values=None, numRows=rows,
            stride0=logits.stride(0), stride1=1, k=2048, stable=stable,
        )  # fmt: skip

    return run


def main():
    args = sys.argv[1:]
    picked = range(len(CASES))
    if args[:1] == ["--cases"]:
        picked = [int(i) for i in args[1].split(",")]
        args = args[2:]
    results = []
    print("rows  width    windows   window_GB  aiter_us  aiter_stable_us  ours_us  speedup  read_floor_us  TB/s  %6.5  %6.9  %floor")
    for rows, width, kind in (CASES[i] for i in picked):
        logits = make_logits("gauss", rows, width)
        ks, ke = make_windows(rows, width, kind)
        window_bytes = float((ke - ks).clamp(min=0).sum()) * 4
        out = torch.empty(rows, 2048, dtype=torch.int32, device="cuda")
        # cold inputs: several logits tiles when one is smaller than the LLC
        tiles = [logits] + [logits.clone() for _ in range(max(0, 2 * LLC_BYTES // logits.numel() // 4))]
        t_aiter = replay_us(lambda: [aiter_fn(t, ks, ke, out, False)() for t in tiles])[0] / len(tiles)
        t_stable = replay_us(lambda: [aiter_fn(t, ks, ke, out, True)() for t in tiles])[0] / len(tiles)
        t_ours = replay_us(lambda: [host.topk_rows(t, ks, ke, out) for t in tiles])[0] / len(tiles)
        floors = [read_floor_fn(t, ks, ke) for t in tiles]
        t_floor = replay_us(lambda: [f() for f in floors])[0] / len(tiles)
        tbs = window_bytes / t_ours / 1e6
        print(
            f"{rows:5d} {width:7d} {kind:9s} {window_bytes / 1e9:9.2f} {t_aiter:9.0f} {t_stable:15.0f} {t_ours:8.0f} {t_aiter / t_ours:7.2f}x "
            f"{t_floor:13.0f} {tbs:5.2f} {100 * tbs / HBM_PLAIN_TBS:5.0f}% {100 * tbs / HBM_NT_TBS:5.0f}% {100 * t_floor / t_ours:6.0f}%"
        )
        results.append(dict(rows=rows, width=width, windows=kind, aiter_us=t_aiter, aiter_stable_us=t_stable, us=t_ours, floor_us=t_floor, tbs=tbs))
        del tiles, logits
    if args:
        json.dump(results, open(args[0], "w"), indent=1)


if __name__ == "__main__":
    main()
