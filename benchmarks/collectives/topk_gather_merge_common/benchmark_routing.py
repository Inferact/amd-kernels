# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""What the gather + merge entry adds to a call, on eight ranks: graph replay of the communicator built through the
entry against one built from the shape package directly (graphs of 100 calls on distinct inputs, MAX over ranks,
min of 15 rounds, each graph timed twice interleaved, us per call), and the host time of the call's reason (min of
15 rounds of 2000, rank 0).
Run with eight visible GPUs: python3 benchmarks/collectives/topk_gather_merge_common/benchmark_routing.py
"""

import importlib
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, K, PAGE, STRIDE, BLOCKS = 8, 2048, 64, 72, 64
ROWS = (12, 65, 192)


def inputs(rank, rows, dev, n=100):
    g = torch.Generator(device=dev).manual_seed(rank * 31 + rows)
    sets = []
    for _ in range(n):
        scores = torch.randn(rows, K, generator=g, device=dev) * 4
        idx = (torch.rand(rows, K, generator=g, device=dev).argsort(1).int() % (PAGE * BLOCKS)).int()
        bt = torch.randint(0, 1 << 20, (rows, BLOCKS), generator=g, device=dev, dtype=torch.int32)
        sets.append((scores, idx, bt, torch.empty(rows * K, dtype=torch.int32, device=dev),
                     torch.empty(rows + 1, dtype=torch.int32, device=dev)))
    return sets


def time_graph(fn, sets, group):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for s in sets:
            fn(*s)
    torch.cuda.synchronize()
    dist.barrier(group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for s in sets:
            fn(*s)
    g.replay()
    torch.cuda.synchronize()
    dist.barrier(group)
    ts = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(4):
            g.replay()
        b.record()
        b.synchronize()
        t = torch.tensor([a.elapsed_time(b) * 1000 / (4 * len(sets))], dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        ts.append(t.item())
    del g
    return min(ts)


def host_us(fn, n=2000):
    best = float("inf")
    for _ in range(15):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t) * 1e6 / n)
    return best


def run(rank, port):
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank)
    from amd_kernels.collectives.topk_gather_merge_common import host

    group, dev = dist.group.WORLD, torch.device("cuda", rank)
    comm = host.topk_gather_merge_comm(rank, group, dev, topk=K)
    direct = importlib.import_module(host._SHAPE_HOSTS[(K, WORLD)]).TopkGatherMerge(rank, group, dev)
    comm.warmup()
    direct.warmup()
    if rank == 0:
        print("| rows | direct us | entry us | added us | reason host us |\n|---|---|---|---|---|", flush=True)
    for rows in ROWS:
        sets = inputs(rank, rows, dev)
        t = {"direct": [], "entry": []}
        for _ in range(2):
            for name, c in (("direct", direct), ("entry", comm)):
                t[name].append(time_graph(lambda *s, c=c: c(*s, PAGE, STRIDE), sets, group))
        d, e = min(t["direct"]), min(t["entry"])
        reason = host_us(lambda s=sets[0]: host.topk_gather_merge_unsupported_reason(comm, *s, PAGE, STRIDE))
        if rank == 0:
            print(f"| {rows} | {d:.2f} | {e:.2f} | {e - d:+.2f} | {reason:.2f} |", flush=True)
        del sets
    torch.cuda.synchronize()
    comm.close()
    direct.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(port,), nprocs=WORLD, join=True)
