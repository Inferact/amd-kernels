# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two ranks: each projects its 3072-column shard and the exchange writes [rows, 6144] on every rank, equal bit for
bit to the two local shard products; queued eager calls with changing rows, a delayed rank (slot reuse), and both
exchange forms (push / pull) at several rows per group."""
import time
import pytest
import torch
import torch.distributed as dist
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720 import bf16_concat_linear, pack_weight
from amd_kernels.gemm.bf16_concat_linear_n3072_k30720.comm import ConcatLinearComm
from tests.multi_gpu import run_on_2_gpus

ROWS = (8, 56, 70, 80, 128, 256, 448, 512, 513, 1024, 1025, 2048)


def worker(rank, errors, exchanges):
    dev = torch.device('cuda', rank)
    g = torch.Generator(device=dev).manual_seed(11)
    full = (torch.randn((6144, 30720), device=dev, generator=g) * 0.02).bfloat16()  # same on every rank
    packed = [pack_weight(full[r * 3072:(r + 1) * 3072]) for r in range(2)]
    del full
    c = ConcatLinearComm(packed[rank], rank, dist.group.WORLD)
    c.warm_up()
    history = []
    for call in range(2 * len(ROWS)):
        m = ROWS[call % len(ROWS)]
        gi = torch.Generator(device=dev).manual_seed(1000 + call)
        taps = [torch.randn((m, 6144), device=dev, generator=gi).bfloat16() for _ in range(5)]
        if rank == call % 2:
            time.sleep(0.01)
        out = torch.full((m, 6144), float('nan'), device=dev, dtype=torch.bfloat16)
        ex = exchanges[call % len(exchanges)] if exchanges else None
        c(taps, out, exchange_config=ex)
        history.append((taps, out))
    torch.cuda.synchronize()
    for taps, out in history:
        want = torch.cat([bf16_concat_linear(taps, p) for p in packed], dim=1)
        if not torch.equal(out, want):
            errors.append(f'rank {rank}: rows {out.shape[0]} differ in {int((out != want).sum())} values')
    c.close()


@pytest.mark.parametrize('exchanges', (None, ((1, 1, True), (2, 3, False), (4, 6, True), (1, 12, False), (8, 2, False), (2, 1, True))))
def test_exchange_rank_skew_and_slot_reuse(exchanges):
    run_on_2_gpus(worker, exchanges, timeout=600)
