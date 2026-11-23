# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the indexer query RoPE + MXFP4 quant.

GPT-J RoPE on dims 64..127 of each 128-dim head (pair (2j, 2j + 1) rotates by cos | sin of ``cos_sin_cache[position]``,
cos = columns 0..31), in fp32 from the bf16 query, the rotated values rounded to bf16. MXFP4 per 32 dims: e8m0 scale
``(bits(max(amax, 6 * 2^-126) * fl32(1 / 6)) + 0x7FFFFF) >> 23`` (2^ceil(log2(amax / 6)), so nothing saturates), values
times ``2^(254 - e8m0)`` rounded to e2m1 with ties to even, two per byte, the even dim in the low nibble. The weights
are ``fp32(w) * weights_scale``. The kernel is bit for bit this.
"""

import torch

SCALE_GROUP = 32


def rope_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """bf16-rounded GPT-J rotation of the last ``rope_dim`` dims of ``x`` (fp32), cos / sin ``[..., rope_dim // 2]``."""
    out = x.clone()
    lo = x.shape[-1] - rope_dim
    even, odd = x[..., lo::2], x[..., lo + 1 :: 2]
    out[..., lo::2] = (even * cos - odd * sin).bfloat16().float()
    out[..., lo + 1 :: 2] = (odd * cos + even * sin).bfloat16().float()
    return out


def mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed u8 ``[..., D / 2]``, e8m0 u8 ``[..., D / 32]``) of fp32 ``x [..., D]``."""
    g = x.reshape(*x.shape[:-1], x.shape[-1] // SCALE_GROUP, SCALE_GROUP)
    amax = g.abs().amax(-1, keepdim=True)
    floor = torch.tensor(6 * 2.0**-126, dtype=torch.float32, device=x.device)
    ratio = torch.maximum(amax, floor) * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    e8m0 = (ratio.view(torch.int32) + 0x7FFFFF) >> 23
    qx = g * ((254 - e8m0) << 23).view(torch.float32)
    a = qx.abs()
    code = sum(
        (a > t) if tie_up is False else (a >= t)
        for t, tie_up in ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
    ).to(torch.int32)
    code = code + 8 * (qx.view(torch.int32) < 0).to(torch.int32)
    code = code.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    packed = (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8)
    return packed, e8m0.reshape(*x.shape[:-1], -1).to(torch.uint8)


def q_rope_mxfp4_quant_ref(
    q: torch.Tensor,  # [n, 32, 128] bf16
    positions: torch.Tensor,  # [n] int64
    cos_sin_cache: torch.Tensor,  # [rows, 64] fp32 or bf16
    weights: torch.Tensor,  # [n, 32] bf16
    weights_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (q_fp4 u8 ``[n, 32, 64]``, e8m0 u8 ``[n, 32, 4]``, weights fp32 ``[n, 32]``)."""
    cs = cos_sin_cache[positions].float()
    half = cs.shape[-1] // 2
    rot = rope_gptj(q.float(), cs[:, None, :half], cs[:, None, half:], cs.shape[-1])
    packed, scale = mxfp4_quant(rot)
    return packed, scale, weights.float() * torch.tensor(weights_scale, dtype=torch.float32, device=q.device)
