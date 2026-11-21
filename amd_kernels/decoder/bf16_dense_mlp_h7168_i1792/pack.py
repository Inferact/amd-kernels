# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Load-time layouts: a (16 rows, 32 K) tile is 1 KB contiguous in MFMA lane order (lane l holds row l % 16 and K
8 (l // 16) .. +8)."""
import torch

from .reference import HIDDEN, INTER

GP = 2  # gate groups (and the same number of up groups) per weight block of the gate_up kernel


def _pack(w: torch.Tensor) -> torch.Tensor:
    n, k = w.shape
    return w.view(n // 16, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(n, k)


def pack_gate_up(w: torch.Tensor) -> torch.Tensor:
    """[3584, 7168] merged gate/up weight -> blocks of GP gate groups followed by the GP up groups of the same
    columns, each group packed; shape [3584, 7168]."""
    assert w.shape == (2 * INTER, HIDDEN) and w.dtype == torch.bfloat16
    blocks = INTER // 16 // GP
    gate = w[:INTER].view(blocks, GP, 16, HIDDEN)
    up = w[INTER:].view(blocks, GP, 16, HIDDEN)
    return _pack(torch.cat([gate, up], dim=1).reshape(2 * INTER, HIDDEN).contiguous())


def pack_down(w: torch.Tensor) -> torch.Tensor:
    """[7168, 1792] down weight -> packed groups, shape [7168, 1792]."""
    assert w.shape == (HIDDEN, INTER) and w.dtype == torch.bfloat16
    return _pack(w)


def pack_rows(x: torch.Tensor, tiles: int, ksteps: int = HIDDEN // 32) -> torch.Tensor:
    """[rows, K] bf16 -> packed row tiles [tiles, K / 32, 64, 8] (rows past ``rows`` are zero)."""
    rows, k = x.shape
    xp = torch.zeros(tiles * 16, k, dtype=x.dtype, device=x.device)
    xp[:rows] = x
    return xp.view(tiles, 16, k // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(tiles, k // 32, 64, 8)


def unpack_rows(xp: torch.Tensor, rows: int) -> torch.Tensor:
    """Inverse of ``pack_rows``: [rows, K]."""
    tiles, ks = xp.shape[:2]
    return xp.view(tiles, ks, 4, 16, 8).permute(0, 3, 1, 2, 4).reshape(tiles * 16, ks * 32)[:rows]
