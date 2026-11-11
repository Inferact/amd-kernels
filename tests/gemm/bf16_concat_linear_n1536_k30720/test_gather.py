# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Four ranks: each projects its 1536-column shard and the gather writes [rows, 6144] on every rank, equal bit for
bit to the four local shard products; queued eager calls with changing rows and a delayed rank (slot reuse)."""
import time
import torch
import torch.distributed as dist
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720 import bf16_concat_linear, pack_weight
from amd_kernels.gemm.bf16_concat_linear_n1536_k30720.comm import ConcatLinearComm
from tests.multi_gpu import run_on_4_gpus


def worker(rank, errors):
    dev = torch.device('cuda', rank)
    g = torch.Generator(device=dev).manual_seed(11)
    full = (torch.randn((6144, 30720), device=dev, generator=g) * 0.02).bfloat16()  # same on every rank
    packed = [pack_weight(full[r * 1536:(r + 1) * 1536]) for r in range(4)]
    c = ConcatLinearComm(packed[rank], rank, dist.group.WORLD)
    c.warm_up()
    history = []
    for call in range(24):
        m = (8, 70, 80, 128, 256, 512, 1024, 2048)[call % 8]
        gi = torch.Generator(device=dev).manual_seed(1000 + call)
        taps = [torch.randn((m, 6144), device=dev, generator=gi).bfloat16() for _ in range(5)]
        if rank == call % 4:
            time.sleep(0.01)
        out = torch.full((m, 6144), float('nan'), device=dev, dtype=torch.bfloat16)
        c(taps, out)
        history.append((taps, out))
    torch.cuda.synchronize()
    for taps, out in history:
        want = torch.cat([bf16_concat_linear(taps, p) for p in packed], dim=1)
        assert torch.equal(out, want), out.shape
    c.close()


def test_gather_rank_skew_and_slot_reuse():
    run_on_4_gpus(worker, timeout=300)
