# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract of the eight-rank exchange: ``out[s] = x_s[me]`` bit for bit (int32 words; the hand-off moves bytes, no
arithmetic, no rounding, tolerance zero). ``barrier`` returns when every rank has entered its call n."""
import torch

WORLD = 8


def exchange_reference(xs, rank):
    """``xs[s]`` is rank ``s``'s [8, n] input (row d goes to rank d): rank ``rank``'s [8, n] output."""
    return torch.stack([xs[s][rank] for s in range(WORLD)])


def payload(replay, call, src, n, device):
    """Deterministic [8, n] int32 input of rank ``src`` for call ``call`` of replay ``replay`` (every rank can build
    every other rank's input, so a test needs no communication to know the expected output)."""
    d = torch.arange(WORLD, device=device, dtype=torch.int64)[:, None]
    j = torch.arange(n, device=device, dtype=torch.int64)[None, :]
    h = (replay + 1) * 2654435761 + (call + 1) * 40503 + (src + 1) * 977 + d * 131 + j * 13
    h = (h ^ (h >> 15)) * 2246822519
    return ((h ^ (h >> 13)) & 0x7FFFFFFF).to(torch.int32)
