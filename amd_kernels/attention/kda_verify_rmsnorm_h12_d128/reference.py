# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference for post-convolution verify with compact accepted-prefix records."""

import torch


def kda_verify_rmsnorm_reference(
    qkv,
    gate,
    beta,
    out_gate,
    alog,
    bias,
    state,
    offsets,
    slots,
    weight,
    correction,
    key_gate,
    out,
    lower_bound,
    eps,
):
    """FP32 recurrence, BF16 rounding before gated RMSNorm. Null rows zero outputs and preserve records."""
    h, d = state.shape[1:3]
    q, k, v = qkv.view(-1, 3, h, d).unbind(dim=1)
    A = alog.float().exp()[:, None]
    cu = offsets.tolist()
    for b, slot in enumerate(slots.tolist()):
        if slot <= 0:
            out[cu[b] : cu[b + 1]].zero_()
            continue
        S = state[slot]
        for pos, t in enumerate(range(cu[b], cu[b + 1])):
            qh, kh = q[t].float(), k[t].float()
            qh = qh * torch.rsqrt(qh.square().sum(-1, keepdim=True) + 1e-6) * d**-0.5
            kh = kh * torch.rsqrt(kh.square().sum(-1, keepdim=True) + 1e-6)
            a = (lower_bound * torch.sigmoid(A * (gate[t].float() + bias.view(h, d)))).exp()
            S = S * a[:, None, :]
            u = (v[t].float() - (S * kh[:, None, :]).sum(-1)) * torch.sigmoid(beta[t].float())[:, None]
            S = S + u[:, :, None] * kh[:, None, :]
            correction[b, :, pos] = u
            key_gate[b, :, pos, :d] = k[t]
            key_gate[b, :, pos, d:] = gate[t]
            x = (S * qh[:, None, :]).sum(-1).bfloat16().float()
            out[t] = (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
                * weight
                * torch.sigmoid(out_gate[t].float())
            )
    return out
