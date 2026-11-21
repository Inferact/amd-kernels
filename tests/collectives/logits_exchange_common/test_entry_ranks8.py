# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The logits exchange entry on eight ranks: a communicator built by the entry gives the bits of one built from the
package (owner mode with empty receivers, all-gather mode; both sides of the push crossover); after the warm-ups
every row count of both modes compiles nothing; a CUDA graph of ``gemm_exchange`` on it imports and allocates nothing
and replays the eager bits; another entry's object and a weight of another shape get reasons."""
import importlib

import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus


def _counts(rows, world, empty=()):
    owners = [r for r in range(world) if r not in empty]
    c = [0] * world
    for i, r in enumerate(owners):
        c[r] = rows // len(owners) + int(i < rows % len(owners))
    return c


def worker(rank, errors):
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.collectives.logits_exchange_common import (logits_exchange_comm,
                                                                logits_exchange_comm_unsupported_reason,
                                                                logits_gemm_exchange_shapes,
                                                                logits_gemm_exchange_unsupported_reason)
    from amd_kernels.collectives.logits_exchange_common import host as entry
    from amd_kernels.decoder.bf16_dense_mlp_common import bf16_dense_mlp_shapes, dense_mlp_workspace

    warm_up.set_auto(True)
    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    keys = [key for key in sorted(logits_gemm_exchange_shapes()) if key[2] == dist.get_world_size(group)]
    if not keys:
        errors.append(f"rank {rank}: no covered shape for {dist.get_world_size(group)} ranks")
    for key in keys:
        n, k, world = key
        host = importlib.import_module(entry._SHAPE_HOSTS[key])
        max_rows = host.RECV_ROWS
        reason = logits_exchange_comm_unsupported_reason(rank, group, dev, n_shard=n, k=k, max_rows=max_rows)
        if reason is not None:
            errors.append(f"rank {rank}: {reason}")
            return
        comm = logits_exchange_comm(rank, group, dev, n_shard=n, k=k, max_rows=max_rows)
        direct = host.RowAllToAll(rank, group, dev, max_rows=max_rows)
        w = (torch.randn(n, k, device=dev, generator=torch.Generator(device=dev).manual_seed(rank)) * 0.02
             ).bfloat16()
        x = torch.randn(max_rows + 7, k, device=dev, generator=torch.Generator(device=dev).manual_seed(99)).bfloat16()
        idx_all = torch.randperm(x.shape[0], device=dev, generator=torch.Generator(device=dev).manual_seed(5))
        comm.warmup()
        direct.warmup()
        cases = [(1, None), (16, None), (17, None), (max_rows, None),
                 (8, _counts(8, world, empty=(0, 1, 3, 4, 5, 6, 7))), (96, _counts(96, world, empty=(1, 6))),
                 (97, _counts(97, world, empty=(0,))), (max_rows, _counts(max_rows, world, empty=(2, 3, 5)))]
        eager = []
        for i, (rows, counts) in enumerate(cases):
            idx = idx_all[:rows].contiguous()
            cs = None if counts is None else torch.tensor(counts, dtype=torch.int32, device=dev)
            recv = rows if counts is None else counts[rank]
            buf = i % host.BUFFERS
            r = logits_gemm_exchange_unsupported_reason(comm, x, w, idx, cs, buf)
            if r is not None:
                errors.append(f"rank {rank}: {r}")
            comm.gemm_exchange(x, w, idx, cs, buf)
            got = comm.recv(recv, buf).clone()
            direct.gemm_exchange(x, w, idx, cs, buf)
            ref = direct.recv(recv, buf).clone()
            torch.cuda.synchronize()
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                errors.append(f"rank {rank}: rows {rows} counts {counts}: entry communicator differs")
            eager.append((idx, cs, recv, buf, got))
        warns = []
        compile_watch.arm(warns.append)
        late = compile_watch.late_compiles()
        for rows in range(1, max_rows + 1):
            idx = idx_all[:rows].contiguous()
            comm.gemm_exchange(x, w, idx, None, rows % host.BUFFERS)
            cs = torch.tensor(_counts(rows, world), dtype=torch.int32, device=dev)
            comm.gemm_exchange(x, w, idx, cs, rows % host.BUFFERS)
        torch.cuda.synchronize()
        if compile_watch.late_compiles() != late or warm_up.late_compiles():
            errors.append(f"rank {rank}: compiles after the warm-up: {warns[:2]}")
        for idx, cs, recv, buf, ref in eager[2:4] + eager[6:]:
            comm.recv(max_rows, buf).zero_()
            torch.cuda.synchronize()
            dist.barrier(group)
            real = importlib.import_module
            importlib.import_module = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("import in capture"))
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    allocs = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
                    comm.gemm_exchange(x, w, idx, cs, buf)
                    allocs = torch.cuda.memory_stats(dev)["allocation.all.allocated"] - allocs
            finally:
                importlib.import_module = real
            dist.barrier(group)
            graph.replay()
            torch.cuda.synchronize()
            if allocs:
                errors.append(f"rank {rank}: {allocs} allocations in capture at {idx.numel()} rows")
            if not torch.equal(comm.recv(recv, buf).view(torch.int16), ref.view(torch.int16)):
                errors.append(f"rank {rank}: graph replay differs from eager at {idx.numel()} rows")
        hidden, inter = min(bf16_dense_mlp_shapes())
        other = dense_mlp_workspace(dev, hidden=hidden, inter=inter)
        idx = idx_all[:8].contiguous()
        r = logits_gemm_exchange_unsupported_reason(other, x, w, idx)
        if r is None or "no kernel for communicator" not in r:
            errors.append(f"rank {rank}: another entry's object: {r}")
        r = logits_gemm_exchange_unsupported_reason(comm, x, w[: n // 2], idx)
        if r is None or "communicator built for" not in r:
            errors.append(f"rank {rank}: weight of another shape: {r}")
        comm.close()
        direct.close()
        r = logits_gemm_exchange_unsupported_reason(comm, x, w, idx)
        if r is None or "closed" not in r:
            errors.append(f"rank {rank}: closed communicator: {r}")


def test_logits_exchange_entry_ranks8():
    run_on_8_gpus(worker, timeout=110)
