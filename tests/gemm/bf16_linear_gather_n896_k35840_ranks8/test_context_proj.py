# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The eight-rank sharded projection + gather + RMSNorm against reference.py: eager and graph calls, repeats."""
import torch
import torch.distributed as dist
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8 import ProjComm, context_proj_gather_norm, host
from amd_kernels.gemm.bf16_linear_gather_n896_k35840_ranks8.reference import (K, N_FULL, N_SHARD, gather_norm_reference,
                                                                              linear_reference)

from tests.gemm.bf16_linear_gather_n896_k35840_ranks8.multi_gpu8 import run_on_8_gpus

ROWS = (12, 40, 65, 80, 192)


def shard_weight(r, dev):
    g = torch.Generator().manual_seed(1000 + r)
    return (torch.randn(N_SHARD, K, generator=g) * 0.02).bfloat16().to(dev)


def worker(rank, errors, rows_list):
    dev = torch.device("cuda", rank)
    comm = ProjComm(rank, dist.group.WORLD, dev)
    w = shard_weight(rank, dev)
    wp = host.pack_weight(w)
    nw = (torch.rand(N_FULL, generator=torch.Generator().manual_seed(5)) + 0.5).bfloat16().to(dev)
    for rows in rows_list:
        g = torch.Generator().manual_seed(rows)
        aux = [(torch.randn(rows, K // 5, generator=g) * 0.4).bfloat16().to(dev) for _ in range(5)]
        x = torch.cat(aux, dim=-1)
        xp = host.packed_rows_buffer(rows, dev)
        host.pack_aux(aux, xp, rows)
        shards = [linear_reference(x, shard_weight(r, dev)) if r != rank else linear_reference(x, w) for r in range(8)]
        ref = gather_norm_reference(shards, nw, 1e-6)

        def check(out, tag):
            cos = torch.nn.functional.cosine_similarity(out.float(), ref.float(), dim=1).min().item()
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            if not (cos >= 0.99999 and rel <= 1e-3):
                errors.append(f"rank {rank} rows {rows} {tag}: cos {cos} rel {rel}")

        out = context_proj_gather_norm(comm, xp, wp, nw, rows)
        torch.cuda.synchronize()
        check(out, "eager")
        first = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = context_proj_gather_norm(comm, xp, wp, nw, rows)
        for it in range(3):
            gout.view(torch.int16).fill_(-1)
            graph.replay()
            torch.cuda.synchronize()
            check(gout, f"graph{it}")
            if it and not torch.equal(gout.view(torch.int16), first.view(torch.int16)):
                errors.append(f"rank {rank} rows {rows}: replay differs from eager")
    comm.close()


def test_context_proj():
    run_on_8_gpus(worker, ROWS, timeout=110)
