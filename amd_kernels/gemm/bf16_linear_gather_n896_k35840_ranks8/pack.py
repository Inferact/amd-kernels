# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Load-time layouts of the MFMA operands: 16 rows x 32 K tile = 1 KB, lane l holds row l % 16, K 8 (l // 16) ..+8."""
import torch

from .reference import K, N_SHARD


def pack_weight(w: torch.Tensor) -> torch.Tensor:
    """[896, 35840] bf16 shard -> the packed layout [56 groups, 1120 K steps, 64 lanes, 8] as [896, 35840]."""
    assert w.shape == (N_SHARD, K) and w.dtype == torch.bfloat16
    return w.view(N_SHARD // 16, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(N_SHARD, K)


def pack_rows(x: torch.Tensor, tiles: int | None = None) -> torch.Tensor:
    """[R, 35840] bf16 rows -> [tiles, 1120, 64, 8] packed row tiles (rows past R are zero)."""
    r = x.shape[0]
    tiles = tiles or (r + 15) // 16
    xp = torch.zeros(tiles * 16, K, dtype=x.dtype, device=x.device)
    xp[:r] = x
    return xp.view(tiles, 16, K // 32, 4, 8).permute(0, 2, 3, 1, 4).contiguous().view(tiles, K // 32, 64, 8)
