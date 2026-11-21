# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Inputs of the eight-rank latent tail at served shapes, identical on every rank (the oracle needs all partials)."""
import torch

K = 3584
N = 896
WORLD = 8
SERVED_ROWS = (1, 12, 40, 80, 192, 256)


def rand(shape, seed, dev, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(dev, torch.bfloat16)


def make(m, seed, dev, rank):
    parts = [rand((m, K), seed * 100 + r, dev, 0.7) for r in range(WORLD)]
    norm_w = (1.0 + rand((K,), seed * 100 + 20, dev, 0.1)).bfloat16()
    w = rand((N, K), seed * 100 + 30 + rank, dev, 0.03)
    shared = rand((m, N), seed * 100 + 40 + rank, dev, 1.0)
    return parts, norm_w, w, shared
