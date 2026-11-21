# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference of the layer tail for one rank of eight (TP8), the contract the kernel is tested against.

``layer_tail`` takes this rank's shards and ``all_reduce(t)``: the ranks' bf16 tensors summed in fp32 in rank order
0..7 and rounded to bf16 (the kernel's reduction; ``gather_all_reduce`` builds it from ``torch.distributed``).

Computation (S rows; bf16 unless noted; every ``bf16()`` is a round-to-nearest-even store):
  o_proj      attn = all_reduce(bf16(core @ w_o^T))                      [S, 7168]
  AttnRes     prefix' = bf16(prefix + attn), or attn on a block-write layer (``reset``); sources = blocks[:, :nblocks]
              and prefix'; a source's weight e_s = exp(logit_s - max), logit_s = <v_s, w * q> rsqrt(mean(v_s^2) + eps);
              mixed = sum_s e_s v_s / sum_s e_s; x = bf16(mixed rsqrt(mean(mixed^2) + out_eps) in_nw)
  router      logits = x @ w_gate^T (fp32), sg = sigmoid; top-16 of sg + bias (key desc, lower expert first); the
              weights are sg of the picks / their sum
  latent      lat = bf16(x @ w_ld^T); MXFP4 per 32 (scale code = ceil(amax * f32(1/6)) as an E8M0 exponent, RNE
              E2M1, saturating)
  experts     per pick: gate, up = the expert's w13 (MXFP4, separated gate | up, E8M0 scales) against the latent
              (fp32 accumulate); v = SiTUv2(gate, up) in fp32 (beta 4, linear beta 25); INTER = MXFP4 of v per 32;
              down = INTER against w2; the pick's bf16(down * weight); a token's picks folded in top-k order with a
              bf16 rounding after every add; this rank's partial
  latent AR   lat_r = all_reduce(partials); ln = bf16(lat_r rsqrt(mean(lat_r^2) + ln_eps) ln_w)
  shared      sgu = bf16(x @ w_sgu^T) (gate | up); h = bf16 SiTU(gate, up) = (2 tanh(g/4))(1 + tanh(g/2))(25 tanh(u/25));
              shp = bf16(h @ w_sd^T)
  up          shp[:, rank shard] = bf16(shp[:, shard] + ln @ w_up_shard^T); out = all_reduce(shp)
Tolerance: cosine >= 0.9999 against the kernel on its own x (the hardware rcp / exp2 and the fp32 accumulation
order differ); the picks may differ on a near tie.
"""

import torch

TP, HIDDEN, LAT, E, TOPK, RI, SI = 8, 7168, 3584, 896, 16, 384, 768
UP_N = HIDDEN // TP
BETA, LBETA = 4.0, 25.0
bf = torch.bfloat16

_FP4 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def bf16(t):
    return t.to(bf).float()


def gather_all_reduce(group=None):
    """``all_reduce`` for the tests: gather, then sum in rank order in fp32."""
    import torch.distributed as dist

    def ar(t):
        parts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, t.contiguous(), group=group)
        acc = parts[0].float()
        for p in parts[1:]:
            acc = acc + p.float()
        return acc.to(bf)

    return ar


def fp4_table(device):
    t = torch.tensor(_FP4, dtype=torch.float32, device=device)
    return torch.cat([t, -t])


def unpack_fp4(packed):
    """uint8 [..., K/2] (low nibble first) -> fp32 [..., K]."""
    tab = fp4_table(packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    return torch.stack([tab[lo], tab[hi]], dim=-1).flatten(-2)


def e8m0(code):
    return torch.exp2(code.float() - 127.0)


def dequant_mxfp4(packed, codes):
    """MXFP4 weight uint8 [..., K/2] + E8M0 codes uint8 [..., K/32] -> fp32 [..., K]."""
    v = unpack_fp4(packed)
    return (v.unflatten(-1, (-1, 32)) * e8m0(codes).unsqueeze(-1)).flatten(-2)


def e8m0_roundup(amax):
    wi = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=amax.device)).view(torch.int32)
    return (((wi + 0x7FFFFF) >> 23) & 0xFF).clamp(max=254)


def quant_fp4(v):
    """fp32 [..., K] -> (dequantized fp32, E8M0 codes [..., K/32]): MXFP4 per 32 as aiter's inline quant."""
    g = v.unflatten(-1, (-1, 32))
    code = e8m0_roundup(g.abs().amax(-1))
    x = g / e8m0(code).unsqueeze(-1)
    tab = fp4_table(v.device)[:8]
    a = x.abs().clamp(max=6.0)
    mid = (tab[1:] + tab[:-1]) / 2
    idx = torch.bucketize(a, mid, right=False)
    # ties to even mantissa: 0.25 -> 0, 0.75 -> 1, 1.25 -> 1.0, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4
    tie = torch.isin(a, mid)
    even = (idx % 2 == 0)
    idx = torch.where(tie & ~even, idx + 1, idx)
    q = tab[idx.clamp(max=7)] * torch.sign(x)
    return (q * e8m0(code).unsqueeze(-1)).flatten(-2), code


def situ_aiter(g, u):
    a = BETA * torch.tanh(g / BETA) * torch.sigmoid(g)
    return a * LBETA * torch.tanh(u / LBETA)


def situ_bf16(g, u):
    g, u = g.float(), u.float()
    return bf16((BETA / 2 * torch.tanh(g / BETA)) * (1 + torch.tanh(g / 2)) * (LBETA * torch.tanh(u / LBETA)))


def attn_res(prefix, delta, blocks, nw, qk, in_nw, nblocks, eps, out_eps):
    """-> (prefix', x). ``delta`` None: the prefix restarts (``prefix`` is the new prefix already)."""
    pre = prefix.float() if delta is None else bf16(prefix.float() + delta.float())
    srcs = torch.cat([blocks[:, :nblocks].float(), pre.unsqueeze(1)], dim=1)
    rstd = torch.rsqrt(srcs.pow(2).mean(-1) + eps)
    logit = (srcs * (nw.float() * qk.float())).sum(-1) * rstd
    e = torch.exp(logit - logit.amax(-1, keepdim=True))
    mixed = (e.unsqueeze(-1) * srcs).sum(1) / e.sum(-1, keepdim=True)
    x = bf16(mixed * torch.rsqrt(mixed.pow(2).mean(-1, keepdim=True) + out_eps) * in_nw.float())
    return pre.to(bf), x.to(bf)


def route(x, w_gate, bias):
    sg = torch.sigmoid(x.float() @ w_gate.float().t())
    key = sg + bias.float()
    ids = torch.topk(key, TOPK, dim=-1, sorted=True).indices
    wts = sg.gather(-1, ids)
    return ids, wts / wts.sum(-1, keepdim=True)


def moe(x, w_gate, bias, w_ld, w_sgu, w_sd, w13, w13s, w2, w2s, ln_w, w_up, rank, all_reduce, ln_eps=1e-5):
    """The latent MoE tail on the input row ``x``. ``w13`` uint8 [E, 2 RI, LAT/2], ``w13s`` uint8 [E, 2 RI, LAT/32],
    ``w2`` uint8 [E, LAT, RI/2], ``w2s`` uint8 [E, LAT, RI/32] (natural layouts); ``w_up`` this rank's [896, LAT]."""
    ids, wts = route(x, w_gate, bias)
    lat, _ = quant_fp4(bf16(x.float() @ w_ld.float().t()))
    s = x.size(0)
    y = torch.zeros(s, LAT, device=x.device)
    contrib = torch.zeros(s, TOPK, LAT, device=x.device)
    for t in range(s):
        for k in range(TOPK):
            e = int(ids[t, k])
            gu = dequant_mxfp4(w13[e], w13s[e]) @ lat[t]
            inter, _ = quant_fp4(situ_aiter(gu[:RI], gu[RI:]))
            d = dequant_mxfp4(w2[e], w2s[e]) @ inter
            contrib[t, k] = bf16(d * wts[t, k])
    for k in range(TOPK):
        y = bf16(y + contrib[:, k])
    lat_r = all_reduce(y.to(bf)).float()
    ln = bf16(lat_r * torch.rsqrt(lat_r.pow(2).mean(-1, keepdim=True) + ln_eps) * ln_w.float())
    gu = bf16(x.float() @ w_sgu.float().t())
    h = situ_bf16(gu[:, :SI], gu[:, SI:])
    shp = bf16(h @ w_sd.float().t())
    lo = rank * UP_N
    shp[:, lo : lo + UP_N] = bf16(shp[:, lo : lo + UP_N] + ln @ w_up.float().t())
    return all_reduce(shp.to(bf))


def layer_tail(core, w_o, prefix, blocks, ares_nw, ares_qk, in_nw, moe_weights, rank, all_reduce, nblocks, reset=False,
               eps=1e-5, out_eps=1e-5, ln_eps=1e-5):
    """-> (out [S, 7168], prefix' [S, 7168], x). ``moe_weights``: kwargs of ``moe`` after ``x``, minus rank / all_reduce."""
    attn = all_reduce(bf16(core.float() @ w_o.float().t()).to(bf))
    pre, x = attn_res(attn if reset else prefix, None if reset else attn, blocks, ares_nw, ares_qk, in_nw,
                      nblocks, eps, out_eps)
    return moe(x, rank=rank, all_reduce=all_reduce, ln_eps=ln_eps, **moe_weights), pre, x
