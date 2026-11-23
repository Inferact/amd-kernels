# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the mHC seam finisher (``host.mhc_gate_finish``).

The inputs are what ``gemm/mxfp8_postmix_hc4_n5120_k8192`` writes for the new residual ``R'`` of one seam: the collapse
``x1 = bf16(sum_j q_j R'_j)`` and per (row, column block) records of 32 floats whose sums over the blocks are ``mix =
flatten(R') @ fn^T`` (slots 0 - 23), ``sum R'^2`` (24) and ``sum x1^2`` (25). With them the seam's second half is
(``norm/mhc_post_pre_rmsnorm_h5120/reference.py``, same formulas)::

    rstd   = rsqrt(sum R'^2 / 20480 + rms_eps)
    pre'   = sigmoid(mix[0:4]  * rstd * s0 + b[0:4]) + hc_pre_eps
    post'  = sigmoid(mix[4:8]  * rstd * s1 + b[4:8]) * hc_post_mult
    comb'  = Sinkhorn(mix[8:24] * rstd * s2 + b[8:24])
    out    = bf16(x1 * rsqrt(sum x1^2 / 5120 + norm_eps) * w)

``out`` and the gates agree with the seam kernel's to a few fp32 ulp (the records are summed in another order), ``out`` to
one bf16 ulp in a few elements: per-row cosine >= ``COS_MIN``. The MXFP8 option quantizes the bf16 ``out``
(``reference.quantize_mxfp8_reference``).
"""

import torch

from amd_kernels.norm.mhc_post_pre_rmsnorm_h5120.reference import (  # noqa: F401
    dequantize_mxfp8, quantize_mxfp8_reference, sinkhorn_reference,
)

N = 4
H = 5120
COS_MIN = 0.99999


def gate_finish_reference(partials, x1, hc_scale, hc_base, norm_weight, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                          hc_post_mult, norm_eps, sinkhorn_repeat=20):
    """(post [m, 4, 1], comb [m, 4, 4], layer_input bf16 [m, H], next_pre [m, 4])."""
    m = partials.shape[0]
    tot = partials.double().sum(1).float()
    mix = tot[:, :24]
    rstd = torch.rsqrt(tot[:, 24:25] / (N * H) + rms_eps)
    pre = torch.sigmoid(mix[:, 0:4] * rstd * hc_scale[0] + hc_base[0:4]) + hc_pre_eps
    post = torch.sigmoid(mix[:, 4:8] * rstd * hc_scale[1] + hc_base[4:8]) * hc_post_mult
    a = (mix[:, 8:24] * rstd * hc_scale[2] + hc_base[8:24]).reshape(m, N, N)
    comb = sinkhorn_reference(a, hc_sinkhorn_eps, sinkhorn_repeat)
    xf = x1.float()
    out = (xf * torch.rsqrt(tot[:, 25:26] / H + norm_eps) * norm_weight.float()).bfloat16()
    return post.reshape(m, N, 1), comb, out, pre
