# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MoE tail's coded link (``lossless_x3``) in both forms, bit for bit: the SDMA phase (``slots4_stage_rows_x3`` +
push + ``sum2_x3_add_rmsnorm_mxfp8``) against the pull phase on rows that take every branch of the code (zeros,
exceptions, raw virtual waves, NaN / Inf / denormals), the pull phase's rank sum against the reference, on 2 ranks,
twice in a row per size (the raw stores of the second call land in a region the first one used)."""

import torch
import torch.distributed as dist
from tests.collectives.pull_collectives_ranks2.test_slots4_reduce_scatter import _codec_rows
from tests.multi_gpu import run_on_2_gpus

from .workload import H, WORLD, rand

EPS = 1e-6


def _expect(errors, what, got, want):
    if not torch.equal(got.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)):
        n = int((got.contiguous().view(torch.uint8) != want.contiguous().view(torch.uint8)).sum())
        errors.append(f"{what}: {n} bytes differ")


def _worker(rank, errors, token_counts):
    from amd_kernels.collectives.pull_collectives_ranks2.reference import slots4_reduce_scatter_reference
    from amd_kernels.collectives.seq_parallel_handoffs_h6144_q4096_ranks2 import (
        SdmaSeqParallelHandoffs, SeqParallelHandoffs, shard_rows)
    from amd_kernels.norm.gemma_add_rmsnorm_h6144.reference import add_rmsnorm_reference

    dev = torch.device("cuda", rank)
    pull = SeqParallelHandoffs(rank, dist.group.WORLD, dev, max(token_counts))
    sdma = SdmaSeqParallelHandoffs(rank, dist.group.WORLD, dev, max(token_counts))
    nw = rand((H,), 3, dev, 0.2)
    for t in token_counts:
        rows = shard_rows(t)
        res = rand((rows, H), t + rank, dev)
        for kind in range(5):
            for rep in range(2):
                xs = [_codec_rows(WORLD * rows, p, dev, kind, rep) for p in range(WORLD)]
                tag = f"rank {rank} t {t} kind {kind} call {rep}"
                a = pull.ffn_norm_gather(xs[rank], res, nw, EPS)
                a = [x.clone() for x in a]
                b = sdma.ffn_norm_gather(xs[rank], res, nw, EPS, between=lambda: torch.cuda._sleep(20_000 * (1 + rank)))
                torch.cuda.synchronize()
                for k, name in enumerate(("codes", "scales", "residual")):
                    _expect(errors, f"{tag} sdma vs pull {name}", b[k], a[k])
                want = slots4_reduce_scatter_reference([x[0] for x in xs], [x[1] for x in xs], rank)
                _expect(errors, f"{tag} residual vs reference", a[2], add_rmsnorm_reference(want, res, nw, EPS)[1])
                del a, b, xs
        torch.cuda.empty_cache()
    sdma.drain()


def test_coded_moe_tail_sdma_vs_pull():
    run_on_2_gpus(_worker, [4096, 4097, 8192, 16385], timeout=1500)
