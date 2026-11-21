# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Inputs of the eight-rank AttnRes all-reduce at served shapes, identical on every rank (the oracle needs all partials)."""
import torch

H = 7168
WORLD = 8
# (has prefix, num_blocks, block_write_idx, output norm): a pre-attention call of a write layer, a plain call with a
# full bank, the final call (no output norm), the embedding call (no prefix, empty bank, write), a non-write MLP call
CASES = (
    (True, 3, -1, True),
    (True, 8, -1, False),
    (True, 4, 4, True),
    (False, 0, 0, True),
    (False, 5, -1, True),
    (True, 1, -1, True),
)
SERVED_ROWS = (1, 12, 24, 40, 65, 80, 128, 192, 256)


def rand(shape, seed, dev, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(dev, torch.bfloat16)


def make(m, seed, dev, nblocks=8):
    parts = [rand((m, H), seed * 100 + r, dev, 0.7) for r in range(WORLD)]
    prefix = rand((m, H), seed * 100 + 20, dev, 2.0)
    bank = rand((m, nblocks, H), seed * 100 + 21, dev, 2.0)
    norm_w = (1.0 + rand((H,), seed * 100 + 22, dev, 0.1)).bfloat16()
    qk_w = rand((H,), seed * 100 + 23, dev, 0.3)
    out_w = (1.0 + rand((H,), seed * 100 + 24, dev, 0.3)).bfloat16()
    return parts, prefix, bank, norm_w, qk_w, out_w
