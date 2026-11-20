# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract for two-rank bf16 all-reduce, residual add, Gemma norm and optional MXFP8.

The reduction is ``bf16(ftz(ftz(p0) + ftz(p1)))`` in fp32 at every row: each operand and the sum flush denormals to
a zero of their sign, one bf16 rounding. With two operands the sum is commutative, so it equals AITER's one-stage and
two-stage all-reduce at two ranks and does not depend on the row's owner. Residual addition is fp32 and writes a bf16
residual. The norm uses the unrounded fp32 addition, weight offset one and epsilon 1e-6. Its bf16 output must have
per-row cosine >= 0.99999 against the fp64 norm below. MXFP8 always quantizes the bf16-rounded norm output, in blocks
of 32, using the family reference's scale rule. Codes are row-major or (16, 16) tiled; scales use AITER's shuffled
layout. Padding rows are unwritten. Determinism is bitwise.
"""
import torch


def reduce_reference(parts):
    assert len(parts) == 2

    def ftz(v):
        return torch.where(v.abs() < 2.0**-126, v * 0.0, v)

    return ftz(ftz(parts[0].float()) + ftz(parts[1].float())).bfloat16()


def allreduce_norm_reference(parts, residual, weight, eps=1e-6):
    """Return the fp64 normalized value and exact bf16 residual of the reduction."""
    return add_rmsnorm_reference(reduce_reference(parts), residual, weight, eps)


def add_rmsnorm_reference(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(y in fp64, the residual out bf16) of ``x + residual``; ``bf16(y)`` is the bf16 output."""
    s = x.float() + residual.float()
    s64 = s.double()
    eps32 = torch.tensor(eps, dtype=torch.float32).item()
    ss = (s64 * s64).sum(-1, keepdim=True)
    ss = torch.where(ss > torch.finfo(torch.float32).max, torch.full_like(ss, float("inf")), ss)
    y = s64 * torch.rsqrt(ss / 6144 + eps32) * (weight.double() + 1.0)
    return y, s.bfloat16()
