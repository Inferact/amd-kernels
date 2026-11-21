# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight spawned ranks: the lm_head GEMM's publish epilogue into all_to_all_bf16_n20480_ranks8's slot, then its pull.

The received bytes must equal the contiguous GEMM of the mapped rows followed by an RCCL all_to_all (owner rows) or
all_gather (all-gather counts), at the served row counts, dense and random owners, ranks owning 0 rows, all rows on one
owner; slots alternate through graph replays with new inputs and row maps.
"""
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N, K, WORLD = 20480, 7168, 8


def _a2a_reference(y, counts, rank):
    from amd_kernels.collectives.all_to_all_bf16_n20480_ranks8.reference import merge_received_reference
    n = counts[rank]
    recv = torch.empty(WORLD * n * N * 2, dtype=torch.uint8, device=y.device)
    dist.all_to_all_single(recv, y.view(torch.uint8).flatten(), output_split_sizes=[n * N * 2] * WORLD,
                           input_split_sizes=[c * N * 2 for c in counts])
    return merge_received_reference(recv.view(torch.bfloat16).view(WORLD, n, N))


def _ag_reference(y):
    parts = [torch.empty_like(y) for _ in range(WORLD)]
    dist.all_gather(parts, y)
    return torch.cat(parts, dim=1)


def _worker(rank, port):
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10), device_id=dev)
    group = dist.new_group(backend="gloo")
    from amd_kernels.collectives.all_to_all_bf16_n20480_ranks8 import RowAllToAll
    from amd_kernels.gemm.bf16_n20480_k7168 import bf16_gemm_n20480_k7168
    gen = torch.Generator(device=dev).manual_seed(4100 + rank)
    w = (torch.randn(N, K, device=dev, generator=gen) * 0.02).to(torch.bfloat16)
    comm = RowAllToAll(rank, group, dev)
    comm.warmup()
    xg = torch.Generator(device=dev).manual_seed(99)
    x = torch.randn(256, K, device=dev, generator=xg).to(torch.bfloat16)
    x[5, 17] = float("nan")
    x[9, 100] = float("inf")
    patterns = []
    for reqs in (3, 40):
        dense = torch.bincount(torch.arange(reqs) % WORLD, minlength=WORLD).tolist()
        slots = torch.randperm(256, generator=torch.Generator().manual_seed(reqs))[:reqs]
        random = torch.bincount(slots % WORLD, minlength=WORLD).tolist()
        for requests in (dense, random):
            for steps in (1, 4):
                patterns.append([v * steps for v in requests])
    patterns += [[0, 0, 24, 0, 0, 0, 0, 0], [40, 0, 0, 0, 0, 0, 0, 0], [0, 7, 0, 1, 0, 0, 0, 0], [16] * 8, [24] * 8, [32] * 8, [11, 0, 0, 0, 0, 0, 0, 0]]
    for k, counts in enumerate(patterns):
        m = sum(counts)
        cs = torch.tensor(counts, dtype=torch.int32, device=dev)
        n = counts[rank]
        maps = [torch.randperm(256, generator=torch.Generator().manual_seed(1000 * k + j))[:m].to(dev) for j in range(3)]
        outs = [torch.empty(max(n, 1), WORLD * N, dtype=torch.bfloat16, device=dev) for _ in maps]
        refs = [_a2a_reference(bf16_gemm_n20480_k7168(x[rm], w), counts, rank) for rm in maps]
        outs_serial = [torch.empty_like(out) for out in outs]
        ag_out = torch.empty(m, WORLD * N, dtype=torch.bfloat16, device=dev)
        ag_serial = torch.empty_like(ag_out)
        ag_ref = _ag_reference(bf16_gemm_n20480_k7168(x[maps[0]], w))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for j, (rm, out) in enumerate(zip(maps, outs)):
                bf16_gemm_n20480_k7168(x, w, row_indices=rm, push=(comm.send_args(j % 2), cs))
                comm.wait()
                out[:n].copy_(comm.recv(n, j % 2))
            bf16_gemm_n20480_k7168(x, w, row_indices=maps[0], push=(comm.send_args(1), None))
            comm.wait()
            ag_out.copy_(comm.recv(m, 1))
            for j, (rm, out) in enumerate(zip(maps, outs_serial)):
                comm.gemm_exchange(x, w, rm, cs, j % 2, serial_above=1000 if j % 2 else 0)
                out[:n].copy_(comm.recv(n, j % 2))
            comm.gemm_exchange(x, w, maps[0], None, 1, serial_above=0)
            ag_serial.copy_(comm.recv(m, 1))
        for repeat in range(3):
            for out in outs + outs_serial:
                out.fill_(float("nan"))
            ag_out.fill_(float("nan"))
            ag_serial.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            for out, ref in zip(outs, refs):
                assert torch.equal(out[:n].view(torch.int16), ref.view(torch.int16)), (rank, k, repeat, "owner")
            assert torch.equal(ag_out.view(torch.int16), ag_ref.view(torch.int16)), (rank, k, repeat, "gather")
            for out, ref in zip(outs_serial, refs):
                assert torch.equal(out[:n].view(torch.int16), ref.view(torch.int16)), (rank, k, repeat, "serial owner")
            assert torch.equal(ag_serial.view(torch.int16), ag_ref.view(torch.int16)), (rank, k, repeat, "serial gather")
        del graph
    comm.close()
    dist.barrier()
    dist.destroy_process_group()


def test_eight_rank_push():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires eight gfx950 GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)
