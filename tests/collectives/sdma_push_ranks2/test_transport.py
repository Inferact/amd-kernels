# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""SdmaPush on 2 ranks as a chain of phases (the use the transport documents): per round an all-gather push into
region 0, its check, then a reduce-scatter-shaped push into region 1, its check. Distinct bytes per round, the
producer slowed down so the copies would read stale bytes if they started early, more rounds than the signal ring
holds, the sources dropped by the caller right after the push (the transport holds them), the receivers check on
their stream right after each wait. Item sizes below and above the striping threshold and uneven ones, on the default
engine and striped over two engines."""

import pytest
import torch
import torch.distributed as dist
from tests.multi_gpu import run_on_2_gpus

WORLD = 2


def _fill(r, rnd, k):  # rank r's bytes of round rnd, kind k
    return (r * 37 + rnd * 11 + k * 5 + 1) % 251


def _worker(rank, errors, nbytes, rounds, engines):
    from amd_kernels.collectives.sdma_push_ranks2 import RING, SdmaPush

    dev = torch.device("cuda", rank)
    t = SdmaPush(rank, dist.group.WORLD, dev, [WORLD * nbytes, WORLD * nbytes], engines=engines)
    assert rounds > RING
    bad = torch.zeros(1, dtype=torch.int64, device=dev)
    peer = 1 - rank
    for rnd in range(rounds):
        # all-gather: my bytes -> slot `rank` of the peer's region 0
        src = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        torch.cuda._sleep(200_000)  # a slow producer: the copies must not start before it is done
        src.fill_(_fill(rank, rnd, 0))
        src[::4093].fill_(_fill(rank, rnd, 9))  # bytes that differ inside a piece
        seq = t.push(0, [(peer, src.data_ptr(), rank * nbytes, nbytes)], keep=(src,))
        del src
        t.wait(0, seq)
        g = t.region(0).view(WORLD, nbytes)
        want = torch.full((nbytes,), _fill(peer, rnd, 0), dtype=torch.uint8, device=dev)
        want[::4093].fill_(_fill(peer, rnd, 9))
        bad += (g[peer] != want).sum()
        # reduce-scatter shape: block p of my buffer -> slot `rank` of peer p's region 1, two items
        rs = torch.tensor([_fill(rank, rnd, 1 + p) for p in range(WORLD)], dtype=torch.uint8, device=dev)
        rs = rs[:, None].expand(WORLD, nbytes).contiguous()
        torch.cuda._sleep(200_000)
        rs.add_(0)
        half = nbytes // 2 // 16 * 16
        seq = t.push(1, [(peer, rs[peer].data_ptr(), rank * nbytes, half),
                         (peer, rs[peer].data_ptr() + half, rank * nbytes + half, nbytes - half)], keep=(rs,))
        del rs
        t.wait(1, seq)
        r1 = t.region(1).view(WORLD, nbytes)
        bad += (r1[peer] != _fill(peer, rnd, 1 + rank)).sum()
    torch.cuda.synchronize(dev)
    t.drain()
    if bad.item():
        errors.append(f"rank {rank}: {bad.item()} stale or wrong bytes over {rounds} rounds")


@pytest.mark.parametrize("engines", [None, (2, 3)])
@pytest.mark.parametrize("nbytes", [1 << 16, (1 << 20) + 4096 + 16, 3 << 21])
def test_push_chain_ordered_and_reused(nbytes, engines):
    run_on_2_gpus(_worker, nbytes, 40, engines, timeout=600)
