# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Every variant compiles: the barriers and every size and path of the exchange, in compile-only mode (one GPU, no
launch: the kernels need the peers). Run with ``AKR_COMPILE_ONLY=1``."""
import pytest
import torch

from tests import compile_only

from amd_kernels.collectives.handshake_ranks8 import host


@pytest.fixture
def comm():
    if not compile_only.ACTIVE:
        pytest.skip("compile-only mode (AKR_COMPILE_ONLY=1) only: the kernels need 8 ranks")
    dev = torch.device("cuda")
    c = host.Rank8Comm.__new__(host.Rank8Comm)
    c.rank, c.device, c.rot = 0, dev, list(range(0x1000, 0x1000 * 9, 0x1000))
    c.ctr = torch.zeros(1024, dtype=torch.int32, device=dev)
    return c


@pytest.mark.parametrize("variant", ["wave", "lane", "pull", "pullw"])
def test_barrier_compiles(comm, variant):
    host.barrier(comm, variant)


@pytest.mark.parametrize("n", host.SIZES)
def test_exchange_compiles(comm, n):
    x = torch.zeros((8, n), dtype=torch.int32, device=comm.device)
    for path in ("ll", "flag", "pull"):
        if path != "ll" or n <= host.LL_MAX_WORDS:
            host.exchange(comm, x, torch.empty_like(x), path)
