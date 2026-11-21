# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract for the eight-rank all-reduce + AttnRes mix + RMSNorm of Kimi-K3.1 decode (hidden 7168, bf16).

Reduction: the eight rank partials are added in rank order 0..7 in fp32 (no denormal flush) and rounded once to bf16
(the delta). Prefix: ``u = bf16(prefix + delta)`` in fp32 (``u = delta`` without a prefix); ``u`` is the prefix out
and, if ``block_write_idx >= 0``, bank row ``block_write_idx`` (which must be >= ``num_blocks``). Mix: the sources are
bank rows 0..num_blocks-1 and ``u``; each source v has logit ``dot(v, norm_weight * qk_weight) *
rsqrt(mean(v * v) + eps)`` and the mix is the softmax-weighted sum of the sources in fp32. Output: the mix, or with an
output norm ``mix * rsqrt(mean(mix * mix) + out_norm_eps) * out_norm_weight`` (plain RMSNorm, no weight offset),
rounded once to bf16. No quantization. With ``out_packed`` the output rows are stored in the (16 rows x 32 columns)
tile-packed layout of ``pack_tiles``; values are unchanged.

Tolerance of the tests: the delta and the prefix out bit for bit; the output within one bf16 ulp (2^-7 |y|) + 1e-5 max|y| of the
fp64 evaluation ``y`` of the same bf16 inputs (fp32 summation order, exp2 and rsqrt approximations differ), cosine
>= 0.99999 per row; two runs match bit for bit.
"""
import torch


def reduce_reference(parts):
    """bf16 of the fp32 sum of the partials in rank order."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.bfloat16()


def attn_res_reference(delta, prefix, bank, norm_weight, qk_weight, num_blocks, out_norm_weight=None, eps=1e-6,
                       out_norm_eps=1e-6):
    """(y fp64 [rows, 7168], prefix out bf16): the AttnRes mix and norm of the reduced ``delta``."""
    u = delta.float() if prefix is None else prefix.float() + delta.float()
    u = u.bfloat16()
    h = u.shape[-1]
    srcs = torch.cat([bank[:, :num_blocks].double(), u.double()[:, None]], dim=1)
    w1 = norm_weight.double() * qk_weight.double()
    rstd = torch.rsqrt((srcs * srcs).sum(-1) / h + eps)
    logits = (srcs * w1).sum(-1) * rstd
    p = torch.softmax(logits, dim=-1)
    mix = (p[..., None] * srcs).sum(1)
    if out_norm_weight is not None:
        mix = mix * torch.rsqrt((mix * mix).sum(-1, keepdim=True) / h + out_norm_eps) * out_norm_weight.double()
    return mix, u


def allreduce_attnres_rmsnorm_reference(parts, prefix, bank, norm_weight, qk_weight, num_blocks,
                                        block_write_idx=-1, out_norm_weight=None, eps=1e-6, out_norm_eps=1e-6):
    """(y fp64, prefix out bf16, bank out bf16) of the eight partials ``parts`` (bank out: the input bank with row
    ``block_write_idx`` written)."""
    assert len(parts) == 8
    y, u = attn_res_reference(reduce_reference(parts), prefix, bank, norm_weight, qk_weight, num_blocks,
                              out_norm_weight, eps, out_norm_eps)
    bank = bank.clone()
    if block_write_idx >= 0:
        bank[:, block_write_idx] = u
    return y, u, bank


def pack_tiles(x):
    """x [rows, 7168] -> [ceil(rows / 16) * 16, 7168] in the tile-packed layout: the 16 B pack of row r, columns
    8j .. 8j + 7 lies at byte ((r // 16 * 224 + j // 4) * 64 + j % 4 * 16 + r % 16) * 16 (padding rows are zeros)."""
    rows, h = x.shape
    pad = (rows + 15) // 16 * 16
    xp = torch.zeros((pad, h), dtype=x.dtype, device=x.device)
    xp[:rows] = x
    # [row tile, r, pack column group, j % 4, 8 values] -> [row tile, pack column group, j % 4, r, 8 values]
    t = xp.view(pad // 16, 16, h // 32, 4, 8).permute(0, 2, 3, 1, 4)
    return t.contiguous().view(pad, h)


def unpack_tiles(x, rows):
    """The first ``rows`` rows of a tile-packed [ceil(rows / 16) * 16, 7168] tensor (inverse of ``pack_tiles``)."""
    pad, h = x.shape
    t = x.view(pad // 16, h // 32, 4, 16, 8).permute(0, 3, 1, 2, 4)
    return t.contiguous().view(pad, h)[:rows]
