# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four spawned ranks: the lm_head GEMM's publish epilogue into all_to_all_bf16_n50016_ranks4's slot, then its pull.

The received bytes must equal the contiguous GEMM of the mapped rows followed by an RCCL all_to_all of them, at the
target and draft row counts of every batch, dense and random owners, a rank owning 0 rows, all rows on one owner;
slots alternate through graph replays with new inputs, row maps and counts; no compile after the warm-up.
"""
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N, K, WORLD = 50016, 6144, 4


def _rccl_reference(y, counts, rank):
    from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4.reference import merge_received_reference
    n = counts[rank]
    recv = torch.empty(WORLD * n * N * 2, dtype=torch.uint8, device=y.device)
    dist.all_to_all_single(recv, y.view(torch.uint8).flatten(),
                           output_split_sizes=[n * N * 2] * WORLD,
                           input_split_sizes=[c * N * 2 for c in counts])
    return merge_received_reference(recv.view(torch.bfloat16).view(WORLD, n, N))


def _worker(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=10), device_id=dev)
    group = dist.new_group(backend="gloo")
    from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4 import RowAllToAll
    from amd_kernels.gemm.bf16_n50016_k6144 import (bf16_gemm_n50016_k6144, shuffle_weight,
                                                     warmup_bf16_gemm_n50016_k6144)
    gen = torch.Generator(device=dev).manual_seed(4100 + rank)
    w = shuffle_weight((torch.randn(N, K, device=dev, generator=gen) * 0.02).to(torch.bfloat16))
    comm = RowAllToAll(rank, group, dev)
    comm.warmup()
    warmup_bf16_gemm_n50016_k6144(dev, True, "contiguous")  # the references (the same bits on x[row_indices])
    warmup_bf16_gemm_n50016_k6144(dev, True, "publish")
    torch.cuda.synchronize()
    dist.barrier()
    # the same hidden rows on every rank (as the model's), with NaN / inf rows
    xg = torch.Generator(device=dev).manual_seed(99)
    x = (torch.randn(2048, K, device=dev, generator=xg)).to(torch.bfloat16)
    x[5, 17] = float("nan")
    x[9, 100] = float("inf")
    patterns = []
    for reqs in (1, 3, 10, 16, 32, 64, 128, 256):
        dense = torch.bincount(torch.arange(reqs) % WORLD, minlength=WORLD).tolist()
        slots = torch.randperm(256, generator=torch.Generator().manual_seed(reqs))[:reqs]
        random = torch.bincount(slots % WORLD, minlength=WORLD).tolist()
        for requests in (dense, random):
            for steps in (7, 8):
                patterns.append([v * steps for v in requests])
    patterns += [[0, 0, 24, 0], [40, 0, 0, 0], [0, 7, 0, 1]]
    from flydsl.compiler.jit_function import MlirCompiler
    compile_calls = []
    original = MlirCompiler.compile

    def count_compile(*args, **kwargs):
        compile_calls.append(1)
        return original(*args, **kwargs)

    MlirCompiler.compile = count_compile
    try:
        for k, counts in enumerate(patterns):
            m = sum(counts)
            cs = torch.tensor(counts, dtype=torch.int32, device=dev)
            n = counts[rank]
            maps = [torch.randperm(2048, generator=torch.Generator().manual_seed(1000 * k + j))[:m].to(dev)
                    for j in range(3)]
            outs = [torch.empty(n, WORLD * N, dtype=torch.bfloat16, device=dev) for _ in maps]
            refs = [_rccl_reference(bf16_gemm_n50016_k6144(x[rm], w, shuffled=True), counts, rank) for rm in maps]
            for rm, out, ref in zip(maps, outs, refs):
                bf16_gemm_n50016_k6144(x, w, shuffled=True, row_indices=rm, publish=comm.slot())
                comm.pull(m, cs, out)
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rank, k, counts, "eager")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for rm, out in zip(maps, outs):
                    bf16_gemm_n50016_k6144(x, w, shuffled=True, row_indices=rm, publish=comm.slot())
                    comm.pull(m, cs, out)
            for repeat in range(2):
                for out in outs:
                    out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for out, ref in zip(outs, refs):
                    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rank, k, counts, repeat)
            del graph, outs, refs
        # One captured call, new row maps, counts and hidden rows on every replay; the stage path in between
        # (both producers share the slots and the counter).
        m = 256
        rm = torch.arange(m, device=dev)
        cs = torch.tensor([64] * WORLD, dtype=torch.int32, device=dev)
        out = torch.empty(m, WORLD * N, dtype=torch.bfloat16, device=dev)
        xs = torch.empty(m, N, dtype=torch.bfloat16, device=dev)
        out2 = torch.empty_like(out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bf16_gemm_n50016_k6144(x, w, shuffled=True, row_indices=rm, publish=comm.slot())
            comm.pull(m, cs, out)
            comm(xs, cs, out2)
        for k in range(8):
            counts = [m if j == k % WORLD else 0 for j in range(WORLD)] if k % 2 else [16, 48, 112, 80]
            cs.copy_(torch.tensor(counts, dtype=torch.int32))
            rm.copy_(torch.randperm(2048, generator=torch.Generator().manual_seed(77 + k))[:m].to(dev))
            x[:64].copy_(torch.randn(64, K, device=dev, generator=xg).to(torch.bfloat16))
            xs.view(torch.int16).random_(-32768, 32768, generator=gen)
            ref = _rccl_reference(bf16_gemm_n50016_k6144(x[rm], w, shuffled=True), counts, rank)
            ref2 = _rccl_reference(xs.clone(), counts, rank)
            graph.replay()
            torch.cuda.synchronize()
            n = counts[rank]
            assert torch.equal(out[:n].view(torch.int16), ref.view(torch.int16)), (rank, "dynamic", k)
            assert torch.equal(out2[:n].view(torch.int16), ref2.view(torch.int16)), (rank, "dynamic stage", k)
        assert compile_calls == [], len(compile_calls)
    finally:
        MlirCompiler.compile = original
    del graph
    comm.close()
    dist.barrier()
    dist.destroy_process_group()


def test_four_rank_publish_pull():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires four gfx950 GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)
