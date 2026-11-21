# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA spec-verify decode of one rank (12 heads, head dim 128): f_b gate GEMM + conv + delta rule + gated RMSNorm.

Contract (what vLLM's ``kimi_k3_kda_fb_proj`` + AITER ``kda_fused_recurrent_spec_decode`` compute):

- ``proj`` [M, 6288] bf16 is the fused in_proj output (row stride >= 6288): q|k|v at columns 0 / 1536 / 3072 (pre-conv),
  output gate at 4608, f_a at 6144 (128), beta logits at 6272 (12).
- g1 = bf16(f_a @ f_b_weight^T): fp32 accumulate, one rounding to bf16 (the stock GEMM's output dtype).
- per head, token t of sequence n: conv input x (bf16) of q / k / v channels; ``conv = sum_w hist_w * wt_w`` over the
  4 taps in fp32, ``silu`` in fp32 (no bf16 rounding of q / k / v). History = conv_state positions seed..seed+2 of the
  first slot (seed = max(num_accepted - 1, 0)); token t writes x to position 2 + t and, after the last token,
  positions 0 / 1 take the history's taps 1 / 2.
- gate a = exp(lower_bound * sigmoid(exp(A_log) * (g1 + dt_bias))) per key dim; beta = sigmoid(beta logit);
  q, k are L2-normalized with eps 1e-6 (q also scaled by 128 ** -0.5).
- state S [V, K] fp32 read from slot indices[n, seed]; per token: S = S * a (over K); err = beta * (v - S k);
  o = (S + err k^T) q; S += err k^T; S written to slot indices[n, t] when that slot > 0.
- o -> bf16(o * rsqrt(mean(o^2) + eps) * norm_weight * sigmoid(out_gate)).
- a sequence whose seed slot is <= 0 (padding) writes zeros and touches no state.

All math is fp32. Tolerance of the kernel vs this reference: outputs bf16-level (cosine >= 0.9999, max |diff| <= 2
bf16 ulps of the row max), states fp32 rel 1e-4; the kernel is deterministic run to run.
"""

import math

import torch

NH = 12
D = 128
LP = NH * D
COL_GATE = 3 * LP
COL_FA = 4 * LP
COL_BETA = COL_FA + D
PROJ_COLS = COL_BETA + NH + 4
GATE_EPS_QK = 1e-6


def fb_gate(proj, f_b_weight):
    """g1 [M, 1536] bf16."""
    fa = proj[:, COL_FA:COL_FA + D].float()
    return (fa @ f_b_weight.float().t()).bfloat16()


def kda_decode_fb_gate_spec_reference(
    proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens, state_indices,
    num_accepted, norm_weight, norm_eps, out, conv_strides=None,
):
    """Updates ``conv_state`` / ``state`` in place and fills ``out`` [M, 12, 128] bf16.

    ``conv_weight`` [3, 4, 1536] fp32 (width-major), ``conv_state`` [slots, 4608, P] (any strides), ``state``
    [slots, 12, 128, 128] fp32, ``state_indices`` [N, >=T] int32, ``num_accepted`` [N] int32.
    """
    g1 = fb_gate(proj, f_b_weight).float().view(-1, NH, D)
    a_exp = A_log.float().exp().view(1, NH, 1)
    dtb = dt_bias.float().view(1, NH, D)
    gate = torch.exp(lower_bound * torch.sigmoid(a_exp * (g1 + dtb)))
    beta = torch.sigmoid(proj[:, COL_BETA:COL_BETA + NH].float())
    og = torch.sigmoid(proj[:, COL_GATE:COL_GATE + LP].float()).view(-1, NH, D)
    nw = norm_weight.float()
    wt = conv_weight.float()
    scale = D ** -0.5
    cu = cu_seqlens.tolist()
    for n in range(len(cu) - 1):
        t0, t1 = cu[n], cu[n + 1]
        if t1 == t0:
            continue
        seed = max(int(num_accepted[n]) - 1, 0)
        slot0 = int(state_indices[n, seed])
        if slot0 <= 0:
            out[t0:t1] = 0
            continue
        cslot = int(state_indices[n, 0])
        hist = conv_state[cslot, :, seed:seed + 3].float().t().clone()
        hist_bf = conv_state[cslot, :, seed:seed + 3].clone()
        keep = hist_bf[:, 1:3].clone()
        S = state[slot0].clone()
        for t in range(t1 - t0):
            x = proj[t0 + t, :3 * LP]
            xf = x.float()
            taps = torch.cat([hist, xf[None]], 0)
            w = wt.reshape(3, 4, LP).permute(1, 0, 2).reshape(4, 3 * LP)
            acc = taps[0] * w[0] + taps[1] * w[1] + taps[2] * w[2] + taps[3] * w[3]
            y = acc * torch.sigmoid(acc)
            conv_state[cslot, :, 2 + t] = x
            hist = torch.cat([hist[1:], xf[None]], 0)
            q, k, v = (y[i * LP:(i + 1) * LP].view(NH, D) for i in range(3))
            rq = torch.rsqrt((q * q).sum(-1, keepdim=True) + GATE_EPS_QK) * scale
            rk = torch.rsqrt((k * k).sum(-1, keepdim=True) + GATE_EPS_QK)
            qn, kn = q * rq, k * rk
            S = S * gate[t0 + t][:, None, :]
            err = (v - torch.einsum("hvk,hk->hv", S, kn)) * beta[t0 + t][:, None]
            S = S + err[:, :, None] * kn[:, None, :]
            o = torch.einsum("hvk,hk->hv", S, qn)
            rstd = torch.rsqrt((o * o).mean(-1, keepdim=True) + norm_eps)
            out[t0 + t] = (o * rstd * nw * og[t0 + t]).bfloat16()
            slot = int(state_indices[n, t])
            if slot > 0:
                state[slot] = S
        conv_state[cslot, :, 0:2] = keep
    return out
