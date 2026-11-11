# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Five alternated graph A/B rounds, MAX rank, dense and random request slots.

Baseline: RCCL uint8 all_to_all_single + torch permute/reshape, exactly the
row transport of vLLM batch_shard.all_to_all_rows (no vLLM dependency).
Inputs rotate over >=256 MB. No allocations/fills enter the FlyDSL graph.
"""
import socket
import statistics
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4 import RowAllToAll
from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4.reference import N, WORLD, merge_received_reference


def _capture(calls):
    for c in calls[:2]:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    return graph


def _time(graph, calls):
    dist.barrier()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        graph.replay()
    b.record()
    b.synchronize()
    value = torch.tensor(a.elapsed_time(b) * 1000 / (10 * calls), device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def _fmt(values):
    return f"{statistics.median(values):.2f} [{min(values):.2f}-{max(values):.2f}]"


def _worker(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank, device_id=dev)
    group = dist.new_group(backend="gloo")
    comm = RowAllToAll(rank, group, dev)
    comm.warmup()
    generator = torch.Generator(device=dev).manual_seed(777 + rank)
    for random_slots in (False, True):
        for R in (10, 16, 32, 64, 128, 256):
            slots = torch.randperm(256, generator=torch.Generator().manual_seed(1000 * R))[:R] if random_slots else torch.arange(R)
            owners = torch.bincount(slots % WORLD, minlength=WORLD).tolist()
            for steps in (8, 7):
                counts = [o * steps for o in owners]
                cs = torch.tensor(counts, dtype=torch.int32, device=dev)
                n = counts[rank]
                nsets = max(4, min(100, 1_000_000_000 // (R * steps * N * 2)))
                xs = [torch.randint(-32768, 32768, (R * steps, N), device=dev, dtype=torch.int16,
                                    generator=generator).view(torch.bfloat16) for _ in range(nsets)]
                out = torch.empty(n, WORLD * N, dtype=torch.bfloat16, device=dev)
                recv = torch.empty(WORLD * n * N * 2, dtype=torch.uint8, device=dev)
                def baseline(x):
                    dist.all_to_all_single(recv, x.view(torch.uint8).flatten(),
                                           output_split_sizes=[n * N * 2] * WORLD,
                                           input_split_sizes=[c * N * 2 for c in counts])
                    return merge_received_reference(recv.view(torch.bfloat16).view(WORLD, n, N))
                expected = baseline(xs[0])
                comm(xs[0], cs, out)
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
                graphs = [_capture([lambda x=x: baseline(x) for x in xs]),
                          _capture([lambda x=x: comm(x, cs, out) for x in xs])]
                timings = [[], []]
                for _ in range(5):
                    for i in (0, 1):
                        timings[i].append(_time(graphs[i], nsets))
                if rank == 0:
                    faster = all(t > o for t, o in zip(*timings))
                    print(f"R={R} rows/request={steps} random={random_slots} owners={owners} "
                          f"today={_fmt(timings[0])} ours={_fmt(timings[1])} faster5/5={faster}", flush=True)
                del graphs, xs, recv, out, expected
                torch.cuda.empty_cache()
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)
