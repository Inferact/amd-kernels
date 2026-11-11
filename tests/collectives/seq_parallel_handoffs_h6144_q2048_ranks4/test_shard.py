# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``shard``: rank r's ``shard_rows`` rows of every token, zero padded, every token on one rank in rank order;
``SeqParallelHandoffs`` sizes itself for a batch-token limit clamped to ``MIN_TOKENS``..``MAX_TOKENS`` (4 GPUs)."""

import pytest
import torch
import torch.distributed as dist
from amd_kernels.collectives.seq_parallel_handoffs_h6144_q2048_ranks4 import (
    MAX_TOKENS,
    MIN_TOKENS,
    WORLD,
    SeqParallelHandoffs,
    shard,
    shard_rows,
)
from tests.multi_gpu import run_on_4_gpus


@pytest.mark.parametrize("t", [1, 31, 4096, 4097, 8885, 15745, 32768])
def test_shard(t):
    x = torch.randn(t, 8, generator=torch.Generator().manual_seed(t)).bfloat16()
    parts = [shard(x, r) for r in range(WORLD)]
    rows = shard_rows(t)
    assert all(p.shape == (rows, 8) and p.is_contiguous() for p in parts)
    whole = torch.cat(parts)
    assert torch.equal(whole[:t], x) and not whole[t:].any()


def _clamp_worker(rank, errors):
    dev = torch.device("cuda", rank)
    for asked, want in ((100, MIN_TOKENS), (12000, 12000), (10**6, MAX_TOKENS)):
        ph = SeqParallelHandoffs(rank, dist.group.WORLD, dev, asked)
        if (ph.max_tokens, ph.rows) != (want, shard_rows(want)):
            errors.append(f"rank {rank}: max_tokens {asked} -> {ph.max_tokens}, rows {ph.rows}; want {want}")
        del ph
        torch.cuda.empty_cache()


def test_max_tokens_clamped():
    run_on_4_gpus(_clamp_worker, timeout=900)
