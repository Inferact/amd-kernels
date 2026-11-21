# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference of the KDA decode kernel K1 (Kimi-K3 layer, 12 local heads, head dim 128).

One spec-verify step of ``R`` requests with ``L`` tokens each (``S = R L`` rows), from the AttnRes seam to the gated
core output:

1. AttnRes. ``pre = bf16(prefix + delta)`` replaces ``prefix`` (and is stored at ``blocks[:, write_idx]`` of a
   block-write layer). Sources are ``blocks[:, :nb]`` and ``pre``; per source ``rstd = rsqrt(mean(v^2) + eps)``,
   ``logit = sum(v * norm_w * qk_w) * rstd``, softmax over sources, ``mixed = sum_s w_s v_s / sum_s w_s`` (fp32),
   ``x = bf16(mixed * rsqrt(mean(mixed^2) + out_eps) * in_w)``.
2. ``proj = bf16(x @ w_in^T)`` (fp32 accumulate): q | k | v | g (1536 each), f_a (128), beta (12 + 4 pad).
3. Causal conv (width 4, SiLU) over q | k | v with the conv state, output rounded to bf16 per channel; the state
   keeps the window the accepted prefix leaves (``num_acc``) and the step's inputs (bf16); a request whose first state
   slot is the null slot (0) stores nothing.
4. ``g1 = bf16(f_a @ w_fb^T)``; decay ``= exp(lower_bound * sigmoid(exp(a_log) * (g1 + dt_bias)))``; q, k l2-normed
   (``rsqrt(sum^2 + 1e-6)``, q also scaled by ``128^-0.5``), ``beta = sigmoid(beta)``.
5. Delta-rule recurrence per request and head on the fp32 state ``[128 v, 128 k]`` of slot ``st_idx[r, num_acc-1]``:
   ``S *= decay; v' = (v - S k) beta; S += v' k^T; o = bf16(S q)``; the state after each token is stored to
   ``st_idx[r, j]`` (slots <= 0 skipped). A request whose initial slot is the null slot yields ``o = 0``.
6. ``core = bf16(o * rsqrt(mean(o^2) + onorm_eps) * on_w * sigmoid(g))`` per head.

Tolerance: ``prefix`` / ``blocks`` match bit for bit; conv state and recurrent state match with relative L2 error <= 1e-3,
core with relative L2 error <= 3e-3 and cosine >= 0.9999 (the kernel's sums run in a different order, which flips an
occasional bf16 rounding).
"""

import math

import torch

HIDDEN = 7168
NH = 12
HD = 128
PROJ = NH * HD
CONV_W = 4


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def attn_res(prefix, delta, blocks, norm_w, qk_w, in_w, nb, write_idx, eps, out_eps):
    """Returns x [S, 7168] bf16; updates ``prefix`` / ``blocks`` in place."""
    pre = prefix.float()
    if delta is not None:
        pre = _bf(pre + delta.float())
        prefix.copy_(pre.to(prefix.dtype))
        if write_idx >= 0:
            blocks[:, write_idx].copy_(pre.to(blocks.dtype))
    src = torch.cat([blocks[:, :nb].float(), pre.unsqueeze(1)], dim=1)
    wq = norm_w.float() * qk_w.float()
    rstd = torch.rsqrt(src.pow(2).sum(-1) / HIDDEN + eps)
    logit = (src * wq).sum(-1) * rstd
    ex = torch.exp(logit - logit.max(dim=1, keepdim=True).values)
    mixed = (ex.unsqueeze(-1) * src).sum(1) / ex.sum(1, keepdim=True)
    r = torch.rsqrt(mixed.pow(2).sum(-1, keepdim=True) / HIDDEN + out_eps)
    return (mixed * r * in_w.float()).to(torch.bfloat16)


def kda_decode_k1(
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_in,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    num_acc,
    *,
    qlen,
    nblocks,
    write_idx=-1,
    eps=1e-5,
    out_eps=1e-5,
    onorm_eps=1e-5,
    lower_bound=-5.0,
):
    """Returns ``core`` [S, 1536] bf16; ``prefix``, ``blocks``, ``conv_state`` [slots, 4608, state_len] (any
    stride order) and ``rstate`` [slots, 12, 128, 128] are updated in place."""
    s = prefix.size(0)
    nreq = s // qlen
    sl = conv_state.size(-1)
    x = attn_res(prefix, delta, blocks, ares_nw, ares_qk, in_nw, nblocks, write_idx, eps, out_eps)
    proj = _bf(x.float() @ w_in.float().t())
    qkv_raw, g2, f_a, beta = proj[:, : 3 * PROJ], proj[:, 3 * PROJ : 4 * PROJ], proj[:, 4 * PROJ : 4 * PROJ + HD], proj[
        :, 4 * PROJ + HD : 4 * PROJ + HD + NH
    ]
    cw = conv_w.float().view(3 * PROJ, CONV_W)
    qkv = torch.empty_like(qkv_raw)
    for r in range(nreq):
        acc = int(num_acc[r]) - 1
        slot_c = int(st_idx[r, 0])
        window = conv_state[max(slot_c, 0), :, acc : acc + CONV_W - 1].float()
        old = conv_state[max(slot_c, 0), :, acc + 1 : acc + 1 + sl - qlen].float().clone()
        xin = qkv_raw[r * qlen : (r + 1) * qlen].t()
        seq = torch.cat([window, xin], dim=1)
        o = sum(cw[:, q : q + 1] * seq[:, q : q + qlen] for q in range(CONV_W))
        qkv[r * qlen : (r + 1) * qlen] = _bf(o * torch.sigmoid(o)).t()
        if slot_c > 0:
            conv_state[slot_c] = torch.cat([old, xin], dim=1).to(conv_state.dtype)
    q, k, v = (t.view(s, NH, HD) for t in qkv.split(PROJ, dim=-1))
    qn = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6) * HD**-0.5
    kn = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
    g1 = _bf(f_a @ w_fb.float().t()).view(s, NH, HD) + dt_bias.float().view(NH, HD)
    decay = torch.exp(lower_bound * torch.sigmoid(torch.exp(a_log.float()).view(1, NH, 1) * g1))
    bt = torch.sigmoid(beta)
    out = torch.zeros(s, NH, HD, device=prefix.device)
    for r in range(nreq):
        acc = int(num_acc[r]) - 1
        slot0 = int(st_idx[r, acc])
        if slot0 <= 0:
            continue
        state = rstate[slot0].clone()
        for j in range(qlen):
            t = r * qlen + j
            state = state * decay[t].unsqueeze(1)
            p = (state * kn[t].unsqueeze(1)).sum(-1)
            vn = (v[t] - p) * bt[t].unsqueeze(-1)
            state = state + vn.unsqueeze(-1) * kn[t].unsqueeze(1)
            out[t] = _bf((state * qn[t].unsqueeze(1)).sum(-1))
            slot = int(st_idx[r, j])
            if slot > 0:
                rstate[slot] = state
    ms = out.pow(2).mean(-1, keepdim=True)
    y = out * torch.rsqrt(ms + onorm_eps) * on_w.float() * torch.sigmoid(g2.view(s, NH, HD))
    return y.reshape(s, PROJ).to(torch.bfloat16)
