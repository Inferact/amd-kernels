# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four spawned ranks: raw bits, dynamic GPU counts, changing slots, graphs.

No sampling arithmetic occurs here. Exact byte exchange proves equality for
all processor parameters and seeds, including NaNs and structured masks.
"""
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4.reference import N, WORLD, merge_received_reference


def _rccl_reference(x, counts, rank):
    n = counts[rank]
    recv = torch.empty(WORLD * n * N * 2, dtype=torch.uint8, device=x.device)
    dist.all_to_all_single(recv, x.view(torch.uint8).flatten(),
                          output_split_sizes=[n * N * 2] * WORLD,
                          input_split_sizes=[c * N * 2 for c in counts])
    return merge_received_reference(recv.view(torch.bfloat16).view(WORLD, n, N))


def _worker(rank, port):
    torch.set_num_threads(8)
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD,
                            timeout=timedelta(minutes=3), device_id=dev)
    group = dist.new_group(backend="gloo")
    from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4 import RowAllToAll
    comm = RowAllToAll(rank, group, dev)
    generator = torch.Generator(device=dev).manual_seed(17001 + rank)
    # Dense slots, odd counts, unequal owners, empty receivers, all on one
    # owner. 7 and 8 rows per request exercise both draft and target paths.
    patterns = []
    for reqs in (1, 3, 10, 16, 31, 32, 64, 128, 256):
        dense = torch.bincount(torch.arange(reqs) % WORLD, minlength=WORLD).tolist()
        slots = torch.randperm(256, generator=torch.Generator().manual_seed(reqs))[:reqs]
        random = torch.bincount(slots % WORLD, minlength=WORLD).tolist()
        for requests in (dense, random):
            for steps in (7, 8):
                patterns.append([v * steps for v in requests])
    patterns += [[0, 0, 0, 0], [0, 0, 512, 0], [8, 0, 0, 0], [0, 8, 0, 0]]
    comm.warmup()
    # Start close to signed wrap. All flags must match the counter before
    # the first exchange: otherwise a peer could appear far ahead.
    from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4 import kernel
    from amd_kernels.collectives.all_to_all_bf16_n50016_ranks4.host import _run
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from amd_kernels.common import buffer_ops
    @flyc.kernel(name="test_row_exchange_init_wrap", known_block_size=[64, 1, 1])
    def init_flag(arg_flag: fx.Int64):
        fr = buffer_ops.create_buffer_resource_from_addr(arg_flag, num_records_bytes=4)
        buffer_ops.buffer_store(fx.Int32(2147483645), fr, 0, mask=fx.thread_idx.x == 0, cache_modifier=0x11)
    @flyc.jit
    def launch_test_row_exchange_init_wrap(arg_flag: fx.Int64, stream: fx.Stream):
        init_flag(arg_flag).launch(grid=(1,1,1), block=(64,1,1), stream=stream)
    _run(launch_test_row_exchange_init_wrap, comm.flags[rank], torch.cuda.current_stream())
    comm.counter.fill_(2147483645)
    torch.cuda.synchronize()
    dist.barrier()
    from flydsl.compiler.jit_function import MlirCompiler
    compile_calls = []
    original = MlirCompiler.compile
    def count_compile(*args, **kwargs):
        compile_calls.append(1)
        return original(*args, **kwargs)
    MlirCompiler.compile = count_compile
    try:
        for k, counts in enumerate(patterns):
            cs = torch.tensor(counts, device=dev, dtype=torch.int32)
            n = counts[rank]
            # All bf16 bit patterns: signed zero, subnormals, +/-inf, all
            # NaN payloads. Integer generation avoids canonicalizing NaNs.
            sets = [torch.randint(-32768, 32768, (sum(counts), N), dtype=torch.int16,
                                  device=dev, generator=generator).view(torch.bfloat16) for _ in range(3)]
            outs = [torch.empty(n, WORLD * N, device=dev, dtype=torch.bfloat16) for _ in sets]
            refs = [_rccl_reference(x, counts, rank) for x in sets]
            for x, out, ref in zip(sets, outs, refs):
                comm(x, cs, out)
                torch.cuda.synchronize()
                assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rank, k, "eager")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for x, out in zip(sets, outs):
                    comm(x, cs, out)
            for repeat in range(3):
                for out in outs:
                    out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for out, ref in zip(outs, refs):
                    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rank, k, repeat)
            del graph, sets, outs, refs
        # Same captured addresses, new counts and inputs on every replay.
        # This detects a host plan or sequence frozen into the graph.
        x = torch.empty(80, N, dtype=torch.bfloat16, device=dev)
        out = torch.empty(80, WORLD * N, dtype=torch.bfloat16, device=dev)
        cs = torch.tensor([20] * WORLD, dtype=torch.int32, device=dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            comm(x, cs, out)
        for k in range(16):
            counts = ([80 if j == k % WORLD else 0 for j in range(WORLD)] if k % 2
                      else [8, 16, 24, 32])
            cs.copy_(torch.tensor(counts, dtype=torch.int32))
            x.view(torch.int16).random_(-32768, 32768, generator=generator)
            ref = _rccl_reference(x, counts, rank)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:counts[rank]].view(torch.int16), ref.view(torch.int16)), (rank, "dynamic", k)
        assert compile_calls == [], compile_calls
    finally:
        MlirCompiler.compile = original
    del graph
    comm.close()
    dist.barrier()
    dist.destroy_process_group()


def test_four_rank_exchange():
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires four gfx950 GPUs")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port,), nprocs=WORLD, join=True)
