# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight ranks, the 16-slot sum folded into the all-reduce: the output is bit-identical to the partial-input call on the
slot sums (``slot_sum_reference``), eager and graph, and matches the fp64 contract."""
import torch
import torch.distributed as dist
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import (
    LatentTailComm,
    allreduce_latent_rmsnorm_up_proj,
    allreduce_slots_rmsnorm_up_proj,
)
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.reference import slot_sum_reference

from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.multi_gpu8 import run_on_8_gpus
from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.test_latent_tail import bits, check
from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.workload import K, N, WORLD, rand

ROWS = (1, 12, 24, 32, 33, 40, 64)


def make_slots(m, seed, dev):
    """Per rank: a slot buffer with padding rows and a shuffled reverse_sorted (same layout as the experts' kernel)."""
    sorted_rows = m * 16 + 37
    g = torch.Generator().manual_seed(seed)
    slots, rs = [], []
    for r in range(WORLD):
        perm = torch.randperm(sorted_rows, generator=g)[: m * 16].to(torch.int32)
        slots.append(rand((sorted_rows, K), seed * 100 + r, dev, 0.3))
        rs.append(perm.to(dev))
    return slots, rs


def worker(rank, errors):
    dev = torch.device("cuda", rank)
    c = LatentTailComm(rank, dist.group.WORLD, dev)
    nw, w = (1.0 + rand((K,), 5, dev, 0.1)).bfloat16(), rand((N, K), 9 + rank, dev, 0.03)
    for m in ROWS:
        slots, rs = make_slots(m, 100 + m, dev)
        parts = [slot_sum_reference(slots[r], rs[r]) for r in range(WORLD)]
        shared0 = rand((m, N), 300 + m + rank, dev, 1.0)
        full = torch.zeros(m, 7168, dtype=torch.bfloat16, device=dev)
        view = full[:, rank * N:(rank + 1) * N]
        view.copy_(shared0)
        allreduce_latent_rmsnorm_up_proj(c, parts[rank], nw, w, view)
        torch.cuda.synchronize()
        want = view.clone()
        check(view, parts, nw, w, shared0, ("partial", m))
        view.copy_(shared0)
        allreduce_slots_rmsnorm_up_proj(c, slots[rank], rs[rank], nw, w, view)
        torch.cuda.synchronize()
        assert torch.equal(bits(view), bits(want)), ("slots vs partial", m)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            allreduce_slots_rmsnorm_up_proj(c, slots[rank], rs[rank], nw, w, view)
        for it in range(2):
            view.copy_(shared0)
            g.replay()
            torch.cuda.synchronize()
            assert torch.equal(bits(view), bits(want)), ("slots graph", m, it)
        for s in slots:
            s.neg_()
        parts = [p.neg() for p in parts]
        view.copy_(shared0)
        g.replay()
        torch.cuda.synchronize()
        check(view, parts, nw, w, shared0, ("slots changed", m))
        if rank == 0:
            print("slots PASS rows", m, flush=True)
    c.close()


def test_slots():
    run_on_8_gpus(worker, timeout=900)
