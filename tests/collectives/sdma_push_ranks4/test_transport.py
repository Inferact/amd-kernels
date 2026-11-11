# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SdmaPush on 4 ranks as a chain of phases (the use the transport documents): per round an all-gather push into
region 0, its check, then a reduce-scatter-shaped push into region 1, its check. Distinct bytes per round, the
producer slowed down so the copies would read stale bytes if they started early, more rounds than the signal ring
holds, the sources dropped by the caller right after the push (the transport holds them), the receivers check on
their stream right after each wait."""

import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_4_gpus

WORLD = 4


def _fill(r, rnd, k):  # rank r's bytes of round rnd, kind k
    return (r * 37 + rnd * 11 + k * 5 + 1) % 251


def _worker(rank, errors, nbytes, rounds):
    from amd_kernels.collectives.sdma_push_ranks4 import RING, SdmaPush

    dev = torch.device("cuda", rank)
    t = SdmaPush(rank, dist.group.WORLD, dev, [WORLD * nbytes, WORLD * nbytes])
    assert rounds > RING
    bad = torch.zeros(1, dtype=torch.int64, device=dev)
    peers = [p for p in range(WORLD) if p != rank]
    for rnd in range(rounds):
        # all-gather: my bytes -> slot `rank` of every peer's region 0
        src = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        torch.cuda._sleep(200_000)  # a slow producer: the copies must not start before it is done
        src.fill_(_fill(rank, rnd, 0))
        seq = t.push(0, [(p, src.data_ptr(), rank * nbytes, nbytes) for p in peers], keep=(src,))
        del src
        t.wait(0, seq)
        g = t.region(0).view(WORLD, nbytes)
        for p in peers:
            bad += (g[p] != _fill(p, rnd, 0)).sum()
        # reduce-scatter shape: block p of my buffer -> slot `rank` of peer p's region 1
        rs = torch.tensor([_fill(rank, rnd, 1 + p) for p in range(WORLD)], dtype=torch.uint8, device=dev)
        rs = rs[:, None].expand(WORLD, nbytes).contiguous()
        torch.cuda._sleep(200_000)
        rs.add_(0)
        seq = t.push(1, [(p, rs[p].data_ptr(), rank * nbytes, nbytes) for p in peers], keep=(rs,))
        del rs
        t.wait(1, seq)
        r1 = t.region(1).view(WORLD, nbytes)
        for p in peers:
            bad += (r1[p] != _fill(p, rnd, 1 + rank)).sum()
    torch.cuda.synchronize(dev)
    t.drain()
    if bad.item():
        errors.append(f"rank {rank}: {bad.item()} stale or wrong bytes over {rounds} rounds")


def test_push_chain_ordered_and_reused():
    run_on_4_gpus(_worker, 1 << 20, 80, timeout=600)
