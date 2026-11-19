# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``mxfp8_bmm_mxfp8_out``.

``y[m, g] = bf16(dequant(x[m, g]) @ dequant(w[g]).T)``: exact dequant (e4m3 x 2^(scale - 127), x per 1 x 32,
w per 32 x 32 block), fp32 sums, one bf16 rounding; then the Triton MXFP8 quantizer of the bf16 ``y`` per 32
columns (E8M0 scale ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)``, e4m3 codes of ``y * 2^(127 - scale)``),
row-major codes [M, G * 1024] and scales [M, G * 32].

Tolerance: the kernel sums in a different order than torch, so its bf16 ``y`` differs from ``bmm_bf16`` in about
1% of the elements (one bf16 ulp): the dequantized codes are compared with this reference's by cosine
(>= 0.9999). Against AITER's FlyDSL batched GEMM (``run_bmm_a8w8_mxfp8``, any tile: scaled 16x16x128 MFMAs in
ascending K) the bf16 ``y`` is equal, so ``quantize_mxfp8`` of its output equals the kernel's codes and scales bit
for bit; against AITER's tuned dispatch (``batched_gemm_a8w8_mxscale_bpreshuffle``, FlyDSL or OPUS by M) too,
except where ``y`` is fp32-subnormal, which OPUS flushes to zero. The tests check both. Two runs match bit for
bit."""

import torch


def dequant(q: torch.Tensor, scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """fp32 of e4m3 ``q`` [..., R, C] with E8M0 ``scale`` [..., R / rows, C / cols]."""
    s = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return q.float() * s.repeat_interleave(rows, -2).repeat_interleave(cols, -1)


def bmm_bf16(x, x_scale, w, w_scale):
    """[M, G * N] bf16 of x [M, G, K] (scales [M, G, K / 32]) times the unshuffled w [G, N, K] (32 x 32 scales)."""
    m, g, k = x.shape
    xd = dequant(x.view(m * g, 1, k), x_scale.reshape(m * g, 1, k // 32), 1, 32).view(m, g, k)
    wd = dequant(w, w_scale, 32, 32)
    return torch.einsum("mgk,gnk->mgn", xd, wd).reshape(m, -1).bfloat16()


def quantize_mxfp8(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-major e4m3 codes [m, C] and E8M0 scale bytes [m, C / 32] of a bf16 [m, C] (the Triton quantizer's
    arithmetic)."""
    m, c = y.shape
    x = y.float().view(m, c // 32, 32)
    amax = x.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    sb = torch.clamp(torch.ceil(torch.log2(amax / 448.0)) + 127.0, 0.0, 254.0)
    q = (x * torch.exp2(127.0 - sb)[..., None]).view(m, c).to(torch.float8_e4m3fn)
    return q, sb.to(torch.uint8)


def mxfp8_bmm_mxfp8_out(x, x_scale, w, w_scale):
    return quantize_mxfp8(bmm_bf16(x, x_scale, w, w_scale))
