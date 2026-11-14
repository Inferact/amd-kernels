# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# From AITER aiter/ops/shuffle.py (shuffle_weight, layout (16, 16) on packed FP4; shuffle_scale; ROCm/aiter edf58c67a).
"""The MXFP4 weight layouts MegaMoeTP reads: ``shuffle_weight`` (codes) and ``shuffle_scale`` (e8m0 scales)."""

import torch


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """Row-major MXFP4 codes [E, N, K / 2] (uint8 or float4_e2m1fn_x2) -> 16 x 16-byte tiles: [E, N / 16, K / 32,
    16 rows, 16 bytes], same shape and dtype as the input."""
    dtype = w.dtype
    u = w.view(torch.uint8)
    E, N, KB = u.shape
    assert N % 16 == 0 and KB % 16 == 0, (N, KB)
    u = u.view(E, N // 16, 16, KB // 16, 16).permute(0, 1, 3, 2, 4).contiguous()
    return u.view(E, N, KB).view(dtype)


def shuffle_scale(s: torch.Tensor) -> torch.Tensor:
    """e8m0 scales [rows, K / 32] (all experts' rows stacked) -> the MFMA scale layout, rows padded to 256 and columns
    to 8 with 0x7F (1.0)."""
    m, n = s.shape
    padded = torch.full(((m + 255) // 256 * 256, (n + 7) // 8 * 8), 0x7F, dtype=torch.uint8, device=s.device)
    padded[:m, :n] = s.view(torch.uint8)
    sm, sn = padded.shape
    out = padded.view(sm // 32, 2, 16, sn // 8, 2, 4).permute(0, 3, 5, 2, 4, 1).contiguous()
    return out.view(sm, sn).view(s.dtype)
