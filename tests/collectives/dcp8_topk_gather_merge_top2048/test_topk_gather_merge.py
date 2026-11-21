# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
import socket
from unittest.mock import patch
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from amd_kernels.collectives.dcp8_topk_gather_merge_top2048.reference import reference
from .workload import K, PAGE, STRIDE, make_block_table, make_plane

WORLD = 8


def check(rank, dev, comm, rows, seed, ties, graph):
    planes = [make_plane(seed, r, rows, ties) for r in range(WORLD)]
    bt = make_block_table(seed, rows)
    scores = planes[rank][0].to(dev)
    idx = planes[rank][1].to(dev)
    btd = bt.to(dev)
    slots = torch.full((rows * K,), -7, dtype=torch.int32, device=dev)
    indptr = torch.full((rows + 1,), -7, dtype=torch.int32, device=dev)
    want_slots, want_indptr = reference(torch.stack([p[0] for p in planes]), planes[rank][1], bt, rank, PAGE, STRIDE)
    runs = []
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            comm(scores, idx, btd, slots, indptr, PAGE, STRIDE)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(3):
                comm(scores, idx, btd, slots, indptr, PAGE, STRIDE)
        for _ in range(2):
            slots.fill_(-7)
            indptr.fill_(-7)
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            runs.append((slots.clone(), indptr.clone()))
    else:
        for _ in range(2):
            slots.fill_(-7)
            indptr.fill_(-7)
            comm(scores, idx, btd, slots, indptr, PAGE, STRIDE)
            torch.cuda.synchronize()
            runs.append((slots.clone(), indptr.clone()))
    for s, p in runs:
        assert torch.equal(p.cpu(), want_indptr), (rank, rows, seed)
        assert torch.equal(s[: want_slots.numel()].cpu(), want_slots), (rank, rows, seed)
    assert torch.equal(runs[0][0][: want_slots.numel()], runs[1][0][: want_slots.numel()])


def _die_with_parent(timeout):
    import ctypes
    import faulthandler
    ctypes.CDLL("libc.so.6").prctl(1, 9)
    faulthandler.dump_traceback_later(timeout, exit=True)


def run(rank, port):
    _die_with_parent(100)
    torch.set_num_threads(4)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=WORLD, rank=rank)
    dev = torch.device("cuda", rank)
    from amd_kernels.collectives.dcp8_topk_gather_merge_top2048 import TopkGatherMerge

    comm = TopkGatherMerge(rank, dist.group.WORLD, dev)
    comm.warmup()
    import flydsl.compiler.jit_function as jf

    with patch.object(jf.MlirCompiler, "compile", side_effect=AssertionError("compile after warmup")):
        for n, rows in enumerate((1, 12, 40, 65, 192, 256)):
            check(rank, dev, comm, rows, n, ties=n % 2 == 0, graph=n % 2 == 1)
            check(rank, dev, comm, rows, 100 + n, ties=n % 2 == 1, graph=n % 2 == 0)
        check(rank, dev, comm, 24, 7, ties=False, graph=True)
        check(rank, dev, comm, 80, 8, ties=True, graph=True)
        check(rank, dev, comm, 12, 9, ties=2, graph=False)
        check(rank, dev, comm, 40, 11, ties=3, graph=False)
        check(rank, dev, comm, 12, 13, ties=5, graph=False)
        check(rank, dev, comm, 65, 14, ties=5, graph=True)
        check(rank, dev, comm, 12, 15, ties=6, graph=False)
        check(rank, dev, comm, 40, 16, ties=6, graph=True)
        check(rank, dev, comm, 24, 12, ties=4, graph=True)
        check(rank, dev, comm, 65, 10, ties=2, graph=True)
    comm.close()
    dist.destroy_process_group()


def test_eight_rank_served_rows_ties_and_graph_replay():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires eight MI355X GPUs")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(run, args=(port,), nprocs=WORLD, join=True)
