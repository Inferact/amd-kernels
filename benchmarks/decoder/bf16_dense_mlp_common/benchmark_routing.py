# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the dense MLP entry adds over a direct call of the routed package: CUDA-graph replay (the graph holds the same
kernels), the host time of an eager call (enqueue only), of the entry's routing (dispatch) and of the call's reason;
min over the rounds of the per-round mean, us per call.

Run: python3 benchmarks/decoder/bf16_dense_mlp_common/benchmark_routing.py
"""
import argparse
import importlib
import time

import torch
import torch.distributed as dist

N_GRAPH, REPS, N_EAGER = 24, 10, 200


def _sync():
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()


def _worst(v):
    if not dist.is_initialized():
        return v
    every = [None] * dist.get_world_size()
    dist.all_gather_object(every, v)
    return [max(e[i] for e in every) for i in range(len(v))]


def timeit(fns, rounds):
    """[(graph us, eager us)] per fn."""
    graphs = []
    for fn in fns:
        fn()
        _sync()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(N_GRAPH):
                fn()
        graphs.append(g)
    gt, et = [[] for _ in fns], [[] for _ in fns]
    for r in range(rounds):
        for i in range(len(fns)) if r % 2 == 0 else range(len(fns) - 1, -1, -1):
            _sync()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(REPS):
                graphs[i].replay()
            b.record()
            b.synchronize()
            gt[i].append(a.elapsed_time(b) * 1e3 / (REPS * N_GRAPH))
            _sync()
            t = time.perf_counter()
            for _ in range(N_EAGER):
                fns[i]()
            et[i].append((time.perf_counter() - t) * 1e6 / N_EAGER)
            torch.cuda.synchronize()
    return [(min(_worst(g)), min(_worst(e))) for g, e in zip(gt, et)]


def host_us(fn, rounds):
    """Host us per call of a host-only function (the reason)."""
    ts = []
    for _ in range(rounds):
        t = time.perf_counter()
        for _ in range(N_EAGER):
            fn()
        ts.append((time.perf_counter() - t) * 1e6 / N_EAGER)
    return min(_worst([min(ts)]))


def _print(rows, res, dispatch, reason):
    (ge, ee), (gd, ed) = res
    print(f"| {rows} | {gd:.2f} | {ge:.2f} | {ed:.2f} | {ee:.2f} | {dispatch:.2f} | {reason:.2f} |", flush=True)


HEADER = ("| rows | graph direct | graph entry | host direct | host entry | dispatch | reason |\n"
          "|---|---|---|---|---|---|---|")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, nargs="+", default=[16, 60, 144])
    p.add_argument("--rounds", type=int, default=15)
    a = p.parse_args()
    from amd_kernels.decoder.bf16_dense_mlp_common import (bf16_dense_mlp, bf16_dense_mlp_shapes,
                                                           bf16_dense_mlp_unsupported_reason, dense_mlp_pack_down,
                                                           dense_mlp_pack_gate_up, dense_mlp_rows_buffer,
                                                           dense_mlp_workspace, host as entry)

    dev = torch.device("cuda")
    for key in sorted(bf16_dense_mlp_shapes()):
        hidden, inter = key
        host = importlib.import_module(entry._SHAPE_HOSTS[key])
        ws = dense_mlp_workspace(dev, hidden=hidden, inter=inter)
        gu = dense_mlp_pack_gate_up((torch.randn(2 * inter, hidden, device=dev) * 0.02).bfloat16())
        dn = dense_mlp_pack_down((torch.randn(hidden, inter, device=dev) * 0.03).bfloat16())
        print(f"dense mlp {key}\n{HEADER}")
        for rows in a.rows:
            xp = dense_mlp_rows_buffer(rows, dev, down=dn).normal_()
            out = torch.empty(rows, hidden, dtype=torch.bfloat16, device=dev)
            _print(rows, timeit([lambda: bf16_dense_mlp(ws, xp, gu, dn, rows, out),
                                 lambda: host.dense_mlp(ws, xp, gu, dn, rows, out)], a.rounds),
                   host_us(lambda: entry._shape_host(tuple(dn.shape)), a.rounds),
                   host_us(lambda: bf16_dense_mlp_unsupported_reason(ws, xp, gu, dn, rows, out), a.rounds))


if __name__ == "__main__":
    main()
