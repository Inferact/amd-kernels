# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Rule 4 in compile-only mode: the served row counts (1-48) at one step count reach exactly the three chain-kernel
variants (16 / 32 / 48 row tiles); a chain kernel serves every temperature and cache layout."""

import pytest
import torch

from amd_kernels.sampling import markov_gumbel_fused_v163840_r512 as mk
from amd_kernels.sampling.markov_gumbel_fused_v163840_r512 import host

K = 3


def test_variants():
    if not mk.available():
        pytest.skip("requires gfx950 with FlyDSL")
    dev = torch.device("cuda")
    host._chain_exe.cache_clear()
    w = torch.zeros(mk.V, mk.RANK, dtype=torch.bfloat16, device=dev)
    ids = torch.zeros(64, dtype=torch.int32, device=dev)
    temp = torch.ones(64, dtype=torch.float32, device=dev)
    seeds = torch.zeros(64, dtype=torch.int64, device=dev)
    cache = torch.zeros(64, K, mk.V, dtype=torch.bfloat16, device=dev)
    for n in range(1, mk.MAX_ROWS + 1):
        zi = lambda *s: torch.zeros(*s, dtype=torch.int64, device=dev)
        mk.markov_gumbel_fused(
            torch.zeros(n, K, mk.V, dtype=torch.bfloat16, device=dev), w, w, ids, zi(n), zi(n, K), zi(n, K), zi(n, K),
            temp, seeds, cache, torch.zeros(mk.scratch_dwords(n), dtype=torch.int32, device=dev),
        )
    info = host._chain_exe.cache_info()
    assert (info.misses, info.currsize) == (3, 3)
