# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank graph replay of I64C7AllGather against the padded RCCL all-gather + index_select it replaces: 100
distinct input sets per graph, five alternated rounds of ten replays, MAX rank, median us.
"""
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from amd_kernels.collectives.all_gather_i64_c7_ranks2 import I64C7AllGather

WORLD = 2


def capture(calls):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for c in calls:
            c()
    torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for c in calls:
            c()
    g.replay()
    torch.cuda.synchronize()
    return g


def time_us(g, n):
    dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        g.replay()
    b.record()
    b.synchronize()
    t = torch.tensor([a.elapsed_time(b) * 1000 / (10 * n)], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def run(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    comm = I64C7AllGather(rank, gloo, dev)
    comm.warmup()
    gen = torch.Generator(device=dev).manual_seed(rank + 7200)
    for R in (10, 16, 32, 40, 48, 52, 64, 128, 256):
        counts = [R // WORLD + int(r < R % WORLD) for r in range(WORLD)]
        n, stride = counts[rank], max(counts)
        src = torch.tensor([i % WORLD * stride + i // WORLD for i in range(R)], device=dev, dtype=torch.int64)
        sets = [(torch.randint(0, 200064, (stride, 7), generator=gen, device=dev, dtype=torch.int64), src, stride,
                 torch.empty(R, 7, device=dev, dtype=torch.int64)) for _ in range(100)]
        gathered = torch.empty(WORLD * stride, 7, dtype=torch.int64, device=dev)

        def baseline(x, src, stride, out):
            dist.all_gather_into_tensor(gathered, x)
            torch.index_select(gathered, 0, src, out=out)

        graphs = [capture([lambda a=a: baseline(*a) for a in sets]),
                  capture([lambda a=a: comm(a[0][:n], *a[1:]) for a in sets])]
        ts = [[], []]
        for _ in range(5):
            for i in (0, 1):
                ts[i].append(time_us(graphs[i], len(sets)))
        if rank == 0:
            print(f"R {R}: baseline {statistics.median(ts[0]):.2f} us, this entry {statistics.median(ts[1]):.2f} "
                  f"[{min(ts[1]):.2f}-{max(ts[1]):.2f}] us", flush=True)
        del graphs
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(port,), nprocs=WORLD, join=True)
