# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-shape inputs: per-rank local top-2048 candidates with ties, -inf padding and dead slots."""
import torch

K = 2048
SERVED_ROWS = (1, 12, 24, 40, 65, 80, 128, 192, 256)
PAGE, STRIDE, BLOCKS = 64, 72, 64


def make_plane(seed, rank, rows, ties):
    g = torch.Generator().manual_seed(seed * 1009 + rank)
    scores = torch.randn(rows, K, generator=g) * 4
    if ties == 2:
        scores = torch.ones(rows, K)
    elif ties:
        scores = scores.round()
    if ties == 6 and rank == 0:
        scores = scores + 100.0
    if ties == 3 or ties == 4:
        special = torch.tensor([0.0, -0.0, float("inf"), float("-inf")] + ([float("nan")] if ties == 4 else []))
        where = torch.rand(rows, K, generator=g) < 0.02
        scores = torch.where(where, special[torch.randint(0, len(special), (rows, K), generator=g)], scores)
    live = torch.full((rows, 1), K) if ties in (5, 6) else torch.randint(K // 3, K + 1, (rows, 1), generator=g)
    dead = torch.rand(rows, K, generator=g).argsort(1) >= live
    scores = scores.masked_fill(dead, float("-inf"))
    idx = torch.rand(rows, K, generator=g).argsort(1)[:, :K].to(torch.int32) % (PAGE * BLOCKS)
    idx = torch.where(dead, -1, idx).to(torch.int32)
    return scores, idx


def make_block_table(seed, rows):
    g = torch.Generator().manual_seed(seed + 77)
    return torch.randint(0, 1 << 20, (rows, BLOCKS), generator=g, dtype=torch.int32)
