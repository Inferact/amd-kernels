# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of the mHC seam (``mhc_post_pre_rmsnorm``).

For ``T`` rows, four residual streams ``R[h]`` of ``H`` columns, the previous sub-layer output ``y`` and the gates of
the previous seam (post ``p``, comb ``c[h][j]``, pre ``q``), in fp32 with bf16 roundings where the tensors are bf16::

    R'_j   = bf16(p_j * y + sum_h c[h][j] * R_h)                       new residual
    mix    = flatten(R') @ fn^T,   rstd = rsqrt(mean(R'^2) + rms_eps)     fn [24, 4H] fp32
    pre'   = sigmoid(mix[0:4]  * rstd * s0 + b[0:4]) + hc_pre_eps
    post'  = sigmoid(mix[4:8]  * rstd * s1 + b[4:8]) * hc_post_mult
    comb'  = Sinkhorn(mix[8:24] * rstd * s2 + b[8:24])                 4 x 4, row softmax + eps, then column / row
                                                                       normalizations (``sinkhorn_repeat`` rounds)
    x1     = bf16(sum_j q_j * R'_j)                                    collapse with the carried pre gate
    out    = bf16(x1 * rsqrt(mean(x1^2) + norm_eps) * w)               the next block's input

Without ``y`` the post-mix is skipped (R' = R); without ``q`` the collapse is stream 0. The optional MXFP8 quantization
reads the bf16 ``out``: per 32 columns the E8M0 byte ``clamp(ceil(log2(amax / 448)) + 127, 0, 254)`` and e4m3 codes of
``out * 2^(127 - byte)``, codes row-major ``[T, H]``, bytes row-major ``[T, H / 32]``.

Tolerance: the kernel's gate projection multiplies bf16 ``R'`` by ``fn`` split into bf16 hi + lo (fp32 accumulation),
its reciprocals are the hardware's, and its square sums are tree-ordered, so gates agree to a few fp32 ulp and ``out`` to
one bf16 ulp in a few elements (the tests check per-row cosine >= 0.99999).
"""

import torch

N = 4


def post_mix_reference(residual, sublayer_out, post_mix, comb_mix):
    """New residual ``[T, 4, H]`` bf16."""
    t, _, h = residual.shape
    f = residual.float()
    p = post_mix.reshape(t, N, 1).float()
    c = comb_mix.reshape(t, N, N).float()
    return (p * sublayer_out.float().unsqueeze(1) + torch.einsum("thj,thk->tjk", c, f)).bfloat16()


def sinkhorn_reference(a: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    p = torch.softmax(a, dim=-1) + eps
    p = p / (p.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        p = p / (p.sum(-1, keepdim=True) + eps)
        p = p / (p.sum(-2, keepdim=True) + eps)
    return p


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()).bfloat16()


def gates_reference(r, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat):
    t = r.shape[0]
    flat = r.reshape(t, -1).float()
    mix = flat @ fn.float().t()
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + rms_eps)
    pre = torch.sigmoid(mix[:, 0:4] * rstd * hc_scale[0] + hc_base[0:4]) + hc_pre_eps
    post = torch.sigmoid(mix[:, 4:8] * rstd * hc_scale[1] + hc_base[4:8]) * hc_post_mult
    a = (mix[:, 8:24] * rstd * hc_scale[2] + hc_base[8:24]).reshape(t, N, N)
    return pre, post.reshape(t, N, 1), sinkhorn_reference(a, hc_sinkhorn_eps, sinkhorn_repeat)


def quantize_mxfp8_reference(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 ``[m, K]`` -> e4m3 codes ``[m, K]`` and E8M0 bytes ``[m, K / 32]``."""
    m, k = y.shape
    blocks = y.float().reshape(m, k // 32, 32)
    amax = blocks.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
    sb = torch.clamp(torch.ceil(torch.log2(amax / 448.0)) + 127.0, 0.0, 254.0)
    codes = (blocks * torch.exp2(127.0 - sb).unsqueeze(-1)).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), sb.to(torch.uint8)


def dequantize_mxfp8(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scale.float() - 127.0).unsqueeze(-1)).reshape(m, k)


def mhc_post_pre_rmsnorm_reference(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult,
    sinkhorn_repeat,
    pre_mix,
    sublayer_out,
    post_mix,
    comb_mix,
    norm_weight,
    norm_eps,
):
    """(residual_out [T, 4, H] bf16, post [T, 4, 1], comb [T, 4, 4], layer_input [T, H] bf16, next_pre [T, 4])."""
    t, _, h = residual.shape
    r = residual if sublayer_out is None else post_mix_reference(residual, sublayer_out, post_mix, comb_mix)
    pre, post, comb = gates_reference(
        r, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult, sinkhorn_repeat
    )
    if pre_mix is None:
        x1 = r[:, 0]
    else:
        x1 = torch.einsum("tj,tjh->th", pre_mix.reshape(t, N).float(), r.float()).bfloat16()
    return r, post, comb, rmsnorm_reference(x1, norm_weight, norm_eps), pre
