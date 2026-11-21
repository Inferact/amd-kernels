# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract: the latent down projection y = x @ w^T and its MXFP4 activation quant, as the routed experts take it.

y: bf16 x [M, 7168] times bf16 w [3584, 7168]^T, summed in fp64 here (fp32 in the kernel, in a fixed split-K order),
rounded once to bf16. Quant: per row, 32 consecutive columns share one e8m0 scale 2^(e - 127), e = the exponent of
fp32(amax * fp32(1/6)) rounded up (ceil(log2(amax / 6))), floored at 92 (amax = max |y| of the 32 bf16 values); the
codes are round-to-nearest-even e2m1 of y / scale, saturating at 6.
That is bit for bit what aiter's `per_1x32_mx_quant_hip(y, quant_dtype=fp4x2, shuffle=False)` returns.
Layout: codes uint8 [M, 1792] (column 2i in the low nibble, 2i + 1 in the high one), scales uint8 [M, 112], row-major.
Finite inputs only.

Tolerance: the kernel's bf16 y differs from the reference's by sum-order roundings only (per row cosine >= 0.99999);
the quant of the kernel's own bf16 y matches `quant_reference` bit for bit, and two runs give the same bits.
"""
import torch

K, N = 7168, 3584
GROUP = 32
SCALE_FLOOR = 92
_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def down_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.double() @ w.double().t()).bfloat16()


def quant_reference(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 of bf16 y [M, 3584]: (codes uint8 [M, 1792], e8m0 bytes uint8 [M, 112])."""
    m, n = y.shape
    v = y.float().view(m, n // GROUP, GROUP)
    amax = v.abs().amax(-1)
    bits = (amax * torch.tensor(1 / 6, dtype=torch.float32, device=y.device)).view(torch.int32)
    e = (((bits + 0x7FFFFF) >> 23)).clamp(min=SCALE_FLOOR)
    scale = (e << 23).int().view(torch.float32)
    q = v / scale[..., None]
    a = q.abs().clamp(max=6.0)
    grid = torch.tensor(_GRID, device=y.device)
    d = (a[..., None] - grid).abs()
    near = d == d.min(-1, keepdim=True).values
    idx = torch.arange(8, device=y.device)
    pri = torch.where(near, (idx % 2) * 16 + idx, torch.full_like(idx, 1000))
    code = pri.argmin(-1) + 8 * torch.signbit(q).int()
    code = code.view(m, n).to(torch.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), e.to(torch.uint8)


def bf16_latent_down_mxfp4_quant_k7168_n3584_reference(x: torch.Tensor, w: torch.Tensor):
    """(codes, scales, y) for x [M, 7168] bf16 and w [3584, 7168] bf16."""
    y = down_reference(x, w)
    q, s = quant_reference(y)
    return q, s, y
