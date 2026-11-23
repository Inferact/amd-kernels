# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the query RoPE + e4m3 quant and the KV RoPE + e4m3 cache insert.

GPT-J RoPE on dims 448..511 of each 512-dim row: pair (2j, 2j + 1) of the 64 RoPE dims rotates by cos | sin of
``cos_sin_cache[position]`` (fp32 ``[rows, 64]``, cos = columns 0..31, sin = 32..63), in fp32 from the bf16 input:
even' = fma(e, cos, -(o * sin)), odd' = fma(o, cos, e * sin). There is no bf16 rounding between RoPE and quant: the e4m3 value is the
saturating (+-448) round-to-nearest-even conversion of ``q * q_scale_inv`` for queries and of ``kv * (1 / kv_scale)``
for the cache row (``1 / kv_scale`` in fp32). Token ``t`` goes to cache row ``slot % block_size`` of page
``slot // block_size``; ``slot < 0`` skips it. Queries are never skipped. Bit for bit what the kernel returns.
"""

import torch


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 fused multiply-add: the exact product in fp64, one fp64 sum, one rounding to fp32."""
    return (a.double() * b.double() + c.double()).float()


def _rope(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    cs = cos_sin_cache[positions]
    shape = [x.shape[0]] + [1] * (x.dim() - 2) + [32]
    cos, sin = cs[:, :32].reshape(shape), cs[:, 32:].reshape(shape)
    x = x.float()
    even, odd = x[..., 448::2], x[..., 449::2]
    out = x.clone()
    out[..., 448::2] = _fma(even, cos, -(odd * sin))
    out[..., 449::2] = _fma(odd, cos, even * sin)
    return out


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def rope_fp8_kv_insert_ref(
    q: torch.Tensor,  # [n, 64, 512] bf16
    kv: torch.Tensor,  # [n, 512] bf16
    kv_cache: torch.Tensor,  # [pages, block_size, 512] e4m3
    slot_mapping: torch.Tensor,  # [n] int64
    positions: torch.Tensor,  # [n] int64
    cos_sin_cache: torch.Tensor,  # [rows, 64] fp32
    kv_scale: torch.Tensor,  # fp32 scalar
    q_scale_inv: torch.Tensor,  # fp32 scalar
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the e4m3 queries ``[n, 64, 512]`` and a copy of the cache with the KV rows inserted."""
    q_fp8 = _e4m3(_rope(q, positions, cos_sin_cache) * q_scale_inv)
    rows = _e4m3(_rope(kv, positions, cos_sin_cache) * (1.0 / kv_scale))
    out = kv_cache.clone()
    flat = out.view(torch.uint8).view(-1, 512)
    ok = slot_mapping >= 0
    flat[slot_mapping[ok]] = rows[ok].view(torch.uint8)
    return q_fp8, out
