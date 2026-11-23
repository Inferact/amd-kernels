# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The linear gather entry on eight ranks: the communicator, packing and op through the entry give the routed
package's bits at the smallest and the largest served row count; after the first call warms the package up, every
row count compiles nothing; a CUDA graph of the entry call imports and allocates nothing and replays the eager bits;
another entry's object and a weight of another shape get reasons."""
import importlib

import torch
import torch.distributed as dist

from tests.multi_gpu8 import run_on_8_gpus


def _inputs(rows, k, layers, dev, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    return [torch.randn(rows, k // layers, device=dev, generator=g).bfloat16() for _ in range(layers)]


def worker(rank, errors):
    from amd_kernels import compile_watch, warm_up
    from amd_kernels.decoder.bf16_dense_mlp_common import bf16_dense_mlp_shapes, dense_mlp_workspace
    from amd_kernels.gemm.bf16_common import (bf16_linear_gather, bf16_linear_gather_blocker,
                                              bf16_linear_gather_shapes, linear_gather,
                                              linear_gather_comm, linear_gather_comm_blocker,
                                              linear_gather_pack_aux, linear_gather_pack_aux_blocker,
                                              linear_gather_pack_weight, linear_gather_rows_buffer)

    warm_up.set_auto(True)
    dev = torch.device("cuda", rank)
    group = dist.group.WORLD
    keys = [key for key in sorted(bf16_linear_gather_shapes()) if key[2] == dist.get_world_size(group)]
    if not keys:
        errors.append(f"rank {rank}: no covered shape for {dist.get_world_size(group)} ranks")
    for key in keys:
        n_shard, k, world = key
        host = importlib.import_module(linear_gather._ROUTES.hosts[key])
        layers, hi = linear_gather._layers(host), linear_gather._max_rows(host)
        reason = linear_gather_comm_blocker(rank, group, dev, n_shard=n_shard, k=k)
        if reason is not None:
            errors.append(f"rank {rank}: {reason}")
            return
        comm = linear_gather_comm(rank, group, dev, n_shard=n_shard, k=k)
        direct = host.ProjComm(rank, group, dev)
        g = torch.Generator(device=dev).manual_seed(1000 + rank)
        w = (torch.randn(n_shard, k, device=dev, generator=g) * 0.01).bfloat16()
        nw = (1 + 0.1 * torch.randn(n_shard * world, device=dev, generator=torch.Generator(device=dev).manual_seed(7))
              ).bfloat16()
        wp = linear_gather_pack_weight(w)
        wd = host.pack_weight(w)
        if not torch.equal(wp, wd):
            errors.append(f"rank {rank}: pack_weight differs")
        eager = {}
        for rows in (1, hi):
            aux = _inputs(rows, k, layers, dev, rows)
            xp = linear_gather_rows_buffer(rows, dev, w_packed=wp)
            xd = host.packed_rows_buffer(rows, dev)
            for r in (linear_gather_pack_aux_blocker(aux, xp, rows, w_packed=wp),
                      bf16_linear_gather_blocker(comm, xp, wp, nw, rows)):
                if r is not None:
                    errors.append(f"rank {rank}: {r}")
            linear_gather_pack_aux(aux, xp, rows, w_packed=wp)
            host.pack_aux(aux, xd, rows)
            out = bf16_linear_gather(comm, xp, wp, nw, rows)
            ref = host.context_proj_gather_norm(direct, xd, wd, nw, rows)
            torch.cuda.synchronize()
            if not torch.equal(xp, xd):
                errors.append(f"rank {rank}: pack_aux differs at {rows} rows")
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                errors.append(f"rank {rank}: output differs from the package at {rows} rows")
            eager[rows] = (xp, out.clone())
        warns = []
        compile_watch.arm(warns.append)
        late = compile_watch.late_compiles()
        for rows in range(1, hi + 1):
            xp = linear_gather_rows_buffer(rows, dev, w_packed=wp).zero_()
            bf16_linear_gather(comm, xp, wp, nw, rows)
        torch.cuda.synchronize()
        if compile_watch.late_compiles() != late or warm_up.late_compiles():
            errors.append(f"rank {rank}: compiles after the warm-up: {warns[:2]}")
        for rows, (xp, ref) in eager.items():
            out = torch.zeros(rows, n_shard * world, dtype=torch.bfloat16, device=dev)
            real = importlib.import_module
            importlib.import_module = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("import in capture"))
            try:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    allocs = torch.cuda.memory_stats(dev)["allocation.all.allocated"]
                    bf16_linear_gather(comm, xp, wp, nw, rows, 1e-5, out)
                    allocs = torch.cuda.memory_stats(dev)["allocation.all.allocated"] - allocs
            finally:
                importlib.import_module = real
            graph.replay()
            torch.cuda.synchronize()
            if allocs:
                errors.append(f"rank {rank}: {allocs} allocations in capture at {rows} rows")
            if not torch.equal(out.view(torch.int16), ref.view(torch.int16)):
                errors.append(f"rank {rank}: graph replay differs from eager at {rows} rows")
        xp = eager[1][0]
        hidden, inter = min(bf16_dense_mlp_shapes())
        other = dense_mlp_workspace(dev, hidden=hidden, inter=inter)
        r = bf16_linear_gather_blocker(other, xp, wp, nw, 1)
        if r is None or "no kernel for communicator" not in r:
            errors.append(f"rank {rank}: another entry's object: {r}")
        r = bf16_linear_gather_blocker(comm, xp, wp[: n_shard // 2], nw, 1)
        if r is None or "communicator built for" not in r:
            errors.append(f"rank {rank}: weight of another shape: {r}")
        comm.close()
        direct.close()
        r = bf16_linear_gather_blocker(comm, xp, wp, nw, 1)
        if r is None or "closed" not in r:
            errors.append(f"rank {rank}: closed communicator: {r}")


def test_linear_gather_entry_ranks8():
    run_on_8_gpus(worker, timeout=110)
