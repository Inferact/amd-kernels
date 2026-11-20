# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank graph replay of SamplerAllGather against vLLM's gather_sampler_output (its Triton pack, an RCCL
all-gather, its Triton unpack): 100 distinct input sets per graph, five alternated rounds of ten replays, MAX rank,
median us.
"""
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from amd_kernels.collectives.all_gather_sampler_i64_c8_ranks2 import SamplerAllGather

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
    from vllm.v1.worker.gpu.sample.batch_shard import _pack_sampler_output_kernel, _unpack_gathered_output_kernel
    comm = SamplerAllGather(rank, gloo, dev)
    comm.warmup()
    gen = torch.Generator(device=dev).manual_seed(rank + 7200)
    for R in (10, 16, 32, 40, 48, 52, 64, 128, 256):
        counts = [R // WORLD + int(r < R % WORLD) for r in range(WORLD)]
        n, stride = counts[rank], max(counts)
        src = torch.tensor([i % WORLD * stride + i // WORLD for i in range(R)], device=dev, dtype=torch.int64)
        cu = torch.arange(0, n * 8 + 1, 8, device=dev, dtype=torch.int32)
        sets = []
        for _ in range(100):
            tok = torch.randint(0, 200064, (n, 8), generator=gen, device=dev, dtype=torch.int64)
            ns = torch.randint(1, 9, (n,), generator=gen, device=dev, dtype=torch.int32)
            nan = torch.randint(0, 10, (n * 8,), generator=gen, device=dev, dtype=torch.int32)
            out = (torch.empty(R, 8, device=dev, dtype=torch.int64),
                   *[torch.empty(R, device=dev, dtype=torch.int32) for _ in range(3)])
            sets.append((tok, ns, 8 - ns, nan, cu, src, stride, *out))
        packed = torch.empty(stride, 11, dtype=torch.int64, device=dev)
        gathered = torch.empty(WORLD * stride, 11, dtype=torch.int64, device=dev)

        def baseline(tok, ns, nr, nan, cu, src, stride, ot, os_, or_, on):
            if n:
                _pack_sampler_output_kernel[(n,)](packed, packed.stride(0), tok, tok.stride(0), ns, nr, nan, cu, 8, 8,
                                                  BLOCK_SIZE=8)
            dist.all_gather_into_tensor(gathered, packed)
            _unpack_gathered_output_kernel[(R,)](gathered, gathered.stride(0), src, ot, os_, or_, on, 8,
                                                 BLOCK_SIZE=8)

        graphs = [capture([lambda a=a: baseline(*a) for a in sets]), capture([lambda a=a: comm(*a) for a in sets])]
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
