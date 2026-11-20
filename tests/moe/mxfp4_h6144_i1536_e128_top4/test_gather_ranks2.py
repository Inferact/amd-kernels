# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks (TP2 sequence-parallel prefill): each prepares its shard, ``gather_mxfp4_moe`` gathers the compact payload
through a two-rank communicator (here RCCL's all-gather behind the ``all_gather(outs, ins)`` contract; to 16384 rows per
rank), and the entry on the gathered input equals the entry on the rank-ordered concatenation of both shards' payloads,
eager and replayed."""
import torch
import torch.distributed as dist

from tests.multi_gpu import run_on_2_gpus

ROWS = (1, 129, 513, 2049, 4096, 8192, 16384)


class _RcclGather:
    """``AllGatherComm``'s contract: every output concatenates the ranks' inputs along the first axis."""

    def __init__(self):
        self.group = dist.new_group(backend="nccl")

    def all_gather(self, outs, ins):
        for out, x in zip(outs, ins):
            dist.all_gather_into_tensor(out, x, group=self.group)


def _weights(dev):
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import MoeWeights, SharedWeights
    from .workload import H, I, make_experts
    torch.cuda.set_device(dev)
    raw, _ = make_experts()
    g = torch.Generator(device=dev).manual_seed(607)

    def fp8(n, k):
        q = (torch.randn((n, k), device=dev, generator=g) * 20).to(torch.float8_e4m3fn).view(torch.uint8)
        return q, torch.randint(118, 124, (n, k // 32), device=dev, dtype=torch.uint8, generator=g)

    shared = SharedWeights.from_row_major(*fp8(2 * I, H), *fp8(H, I))
    gate = torch.randn((128, H), device=dev, generator=g) / H**.5
    bias = torch.randn((128,), device=dev, generator=g) * .01
    return MoeWeights.from_row_major(gate, bias, *raw, shared=shared)


def _worker(rank, errors):
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import (MoeInputCompact, gather_mxfp4_moe, mxfp4_moe,
                                                               prepare_mxfp4_moe)
    from .workload import make_activation
    dev = torch.device("cuda", rank)
    w = _weights(dev)
    comm = _RcclGather()
    for rows in ROWS:
        shards = [make_activation(rows, seed=100 * rows + r) for r in range(2)]
        local = prepare_mxfp4_moe(shards[rank], w)
        payloads = [prepare_mxfp4_moe(x, w).compact() for x in shards]
        expected = MoeInputCompact(*(torch.cat(ts) for ts in zip(*(p.tensors() for p in payloads))))
        gathered = gather_mxfp4_moe(local, comm, world=2)
        torch.cuda.synchronize()
        if not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                   for a, b in zip(gathered.tensors(), expected.tensors())):
            errors.append(f"rank {rank} rows {rows}: gathered payload differs")
        want = mxfp4_moe(expected, w, return_partials=True)
        got = mxfp4_moe(gathered, w, return_partials=True)
        if not all(torch.equal(a, b) for a, b in zip(got, want)):
            errors.append(f"rank {rank} rows {rows}: entry on the gathered payload differs")
        reduced = mxfp4_moe(gathered, w)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = mxfp4_moe(gathered, w)
        replayed.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(replayed, reduced) or not torch.equal(reduced, mxfp4_moe(expected, w)):
            errors.append(f"rank {rank} rows {rows}: replay differs")
        del graph
    try:
        gather_mxfp4_moe(prepare_mxfp4_moe(make_activation(16385, seed=1), w), comm, world=2)
        errors.append("16385 rows per rank accepted")
    except ValueError:
        pass


def test_prepared_gather_two_ranks():
    run_on_2_gpus(_worker, timeout=1800)


def _routed_worker(rank, errors):
    """``routed_only``: the four routed fields (3296 B/row) through the pull ``AllGatherComm``; the routed partials of
    the gathered input equal those of the rank-ordered concatenation."""
    from dataclasses import replace

    from amd_kernels.collectives.pull_collectives_ranks2 import AllGatherComm
    from amd_kernels.moe.mxfp4_h6144_i1536_e128_top4 import (MoeInput, SharedWeights, gather_mxfp4_moe, mxfp4_moe,
                                                               prepare_mxfp4_moe)
    from .workload import make_activation, make_full_shared
    dev = torch.device("cuda", rank)
    w = _weights(dev)
    w = replace(w, shared_weights=SharedWeights.from_row_major(*make_full_shared()[0], tp_rank=rank))
    comm = AllGatherComm(rank, dist.group.WORLD, dev, 16384 * 3296 + 4 * 256)
    comm.warm_up()
    for rows in (1, 129, 1024, 2049, 8192, 16384):
        shards = [make_activation(rows, seed=100 * rows + r) for r in range(2)]
        local = prepare_mxfp4_moe(shards[rank], w)
        payloads = [prepare_mxfp4_moe(x, w).routed() for x in shards]
        expected = MoeInput(*(torch.cat(ts) for ts in zip(*(p.tensors()[:4] for p in payloads))))
        gathered = gather_mxfp4_moe(local, comm, world=2, routed_only=True)
        torch.cuda.synchronize()
        if not gathered.routed_only or not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                               for a, b in zip(gathered.tensors()[:4], expected.tensors()[:4])):
            errors.append(f"rank {rank} rows {rows}: gathered routed payload differs")
        got, none = mxfp4_moe(gathered, w, return_partials=True)
        want, _ = mxfp4_moe(expected, w, return_partials=True)
        torch.cuda.synchronize()
        if none is not None or not torch.equal(got, want):
            errors.append(f"rank {rank} rows {rows}: routed partials of the gathered payload differ")


def test_routed_gather_two_ranks():
    run_on_2_gpus(_routed_worker, timeout=1800)
