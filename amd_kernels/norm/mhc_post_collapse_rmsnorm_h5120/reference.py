# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``mhc_post_collapse_rmsnorm``, the step-end tail of the mHC residual.

With the previous sub-layer output ``y`` [T, H] and the gates ``p`` [T, 4], ``c`` [T, 4, 4] of its seam, the streams
``R`` [T, 4, H], the carried pre gate ``q`` [T, 4] and weight ``w`` [H], in fp32 with bf16 roundings where the tensors
are bf16::

    R'_j = bf16(p_j * y + sum_h c[h][j] * R_h)        absent ``y``: R' = R
    x1   = bf16(sum_j q_j * R'_j)
    out  = bf16(x1 * rsqrt(mean(x1^2) + eps) * w)
    mean = bf16(sum_j R'_j / 4)                       the auxiliary hidden state

Tolerance: the kernel's square sum is tree-ordered and its rsqrt the hardware's, so ``out`` differs from this reference
by one bf16 ulp in a few elements (the tests check per-row cosine >= 0.99999); ``R'``, ``x1`` and ``mean`` match
bit for bit up to the fp32 order of the post-mix sum.
"""

import torch

N = 4


def post_mix_reference(residual, sublayer_out, post_mix, comb_mix):
    t = residual.shape[0]
    p = post_mix.reshape(t, N, 1).float()
    c = comb_mix.reshape(t, N, N).float()
    return (p * sublayer_out.float().unsqueeze(1) + torch.einsum("thj,thk->tjk", c, residual.float())).bfloat16()


def mhc_post_collapse_rmsnorm_reference(residual, pre_mix, norm_weight, eps, sublayer_out=None, post_mix=None,
                                        comb_mix=None):
    """(layer_input [T, H] bf16, residual_out [T, 4, H] bf16, mean [T, H] bf16)."""
    t = residual.shape[0]
    r = residual if sublayer_out is None else post_mix_reference(residual, sublayer_out, post_mix, comb_mix)
    x1 = torch.einsum("tj,tjh->th", pre_mix.reshape(t, N).float(), r.float()).bfloat16().float()
    out = (x1 * torch.rsqrt(x1.square().mean(-1, keepdim=True) + eps) * norm_weight.float()).bfloat16()
    return out, r, r.float().mean(1).bfloat16()
