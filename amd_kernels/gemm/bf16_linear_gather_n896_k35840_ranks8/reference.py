# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of the sharded drafter context projection: ``y = RMSNorm(x @ W^T)`` with W [7168, 35840] split into
eight N-shards of 896 rows (rank r holds rows [896 r, 896 (r + 1))).

``linear_reference``: the fp64 product of the bf16 operands, rounded once to bf16 (a rank's shard [R, 896]). The kernel
accumulates in fp32 (K ascending within a split, splits summed in a fixed order) and rounds once, so it differs from
the reference only in the sum order: per-row cosine >= 0.99999, relative L2 <= 1e-3, deterministic bits.

``gather_norm_reference``: the eight bf16 shards concatenated on the columns, then RMSNorm in fp32 with one rounding:
``bf16(x * rsqrt(mean(x^2) + eps) * weight)`` (the ``ops.rms_norm`` of the stock context_norm); the same tolerance.
"""
import torch

N_SHARD = 896
N_FULL = 7168
K = 35840
WORLD = 8


def linear_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.double() @ w.double().t()).bfloat16()


def gather_norm_reference(shards: list[torch.Tensor], weight: torch.Tensor, eps: float) -> torch.Tensor:
    x = torch.cat(shards, dim=1).float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(var + eps) * weight.float()).bfloat16()
