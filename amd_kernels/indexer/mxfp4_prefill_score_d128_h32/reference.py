# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the FP4 prefill indexer scorer: ``score_rows`` = ``reference_logits`` on every row's window.

Inputs are the FP4 index tensors the serving stack already builds (query rows quantized per 32 elements, key pages staged
in the scorer's preshuffled layout):

- ``q_payload`` uint8 [rows, 32, 64]: head h of a row holds 128 FP4 (E2M1) values, two per byte, low nibble first;
- ``q_scale`` uint8 [rows, 1, 4, 16, 4]: E8M0 exponent (bias 127) of head ``16 * m + l``, values ``32 * g .. 32 * g + 31``
  at ``[row, 0, g, l, m]``;
- ``key_payload`` uint8 [pages, 1, 4, 64, 16]: key ``64 * page + s``, values ``32 * g .. 32 * g + 31`` at
  ``[page, 0, g, s]`` (16 bytes, low nibble first);
- ``key_scale`` uint8 [pages, 1, 4, 64]: E8M0 exponent of those values at ``[page, 0, g, 4 * (s % 16) + s // 16]``;
- ``weights`` bf16 [rows, 32], ``weight_scale`` float; ``ks`` / ``ke`` int32 [rows]: the row's key window.

``logit[r, n] = sum_h weight_scale * weights[r, h] * relu(q[r, h] . k[n])`` for ``ks[r] <= n < ke[r]``, with the dequantized
FP4 operands and the sums in fp32; the kernel stores whole pages, so positions outside the window are unspecified. The kernel matches this to rel. 1e-5 of the
row's largest |logit| (fp32 summation order differs: the kernel sums the heads in the MFMA / cross-lane order of aiter's
scorer) and matches aiter's ``flydsl_pa_mqa_logits_fp4_prefill`` within 2 ulp; two runs match bit for bit.
"""

import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def dequant_fp4(packed: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """FP4 bytes [..., 16 * g_bytes] (low nibble first) with one E8M0 exponent per 16 bytes [..., g] -> fp32 [..., 32 * g]."""
    table = torch.tensor(E2M1, dtype=torch.float32, device=packed.device)
    lo, hi = packed & 15, packed >> 4
    vals = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)
    mag = table[(vals & 7).long()]
    val = torch.where(vals >= 8, -mag, mag)
    g = exponent.shape[-1]
    scale = torch.exp2(exponent.float() - 127.0)
    return (val.reshape(*val.shape[:-1], g, 32) * scale[..., None]).reshape(*val.shape[:-1], 32 * g)


def dequant_queries(q_payload: torch.Tensor, q_scale: torch.Tensor) -> torch.Tensor:
    rows, heads, _ = q_payload.shape
    exp = q_scale[:, 0, :, :, :heads // 16]  # [rows, g=4, l=16, m]
    exp = exp.permute(0, 3, 2, 1).reshape(rows, heads, 4)  # head = 16 * m + l
    return dequant_fp4(q_payload.view(rows, heads, 4, 16).reshape(rows, heads, 64), exp)


def dequant_keys(key_payload: torch.Tensor, key_scale: torch.Tensor) -> torch.Tensor:
    pages = key_payload.shape[0]
    packed = key_payload[:, 0].permute(0, 2, 1, 3).reshape(pages, 64, 64)  # [page, s, g * 16 + byte]
    s = torch.arange(64, device=key_scale.device)
    exp = key_scale[:, 0][:, :, 4 * (s % 16) + s // 16].permute(0, 2, 1)  # [page, s, g]
    return dequant_fp4(packed, exp).reshape(pages * 64, 128)


def reference_logits(q_payload, q_scale, weights, key_payload, key_scale, ks, ke, weight_scale):
    """fp32 logits [rows, pages * 64]; positions outside each row's window are NaN."""
    rows = q_payload.shape[0]
    q = dequant_queries(q_payload, q_scale)
    k = dequant_keys(key_payload, key_scale)
    w = weights.float() * weight_scale
    out = torch.full((rows, k.shape[0]), float("nan"), dtype=torch.float32, device=q.device)
    for r in range(rows):
        lo, hi = int(ks[r]), int(ke[r])
        if hi > lo:
            s = torch.relu(q[r] @ k[lo:hi].T)  # [heads, keys]
            out[r, lo:hi] = w[r] @ s
    return out
