# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch references of MegaMoE TP and SpRsNorm: what "correct" means for the tests and benchmarks.

MegaMoE TP (``mega_moe_tp_ref``, one rank's partial; the layer's output is the sum of the ranks' partials):

- MXFP4 (E2M1, two codes per byte, low nibble first) with one e8m0 scale per 32 values along K. The scale is
  ``2^ceil(log2(amax / 6))`` (round-up), the codes round to nearest even and saturate at 6 (``mxfp4_quant``).
- The input x [M, H] is quantized per row, GEMM1 runs in fp32 on the dequantized operands; gate = columns [0, I),
  up = [I, 2I) of w1 [E, 2I, H]. The activation (``ACTIVATIONS``) runs in fp32 and rounds to bf16, which is quantized
  again to MXFP4 per (token, slot) and multiplied with w2 [E, H, I] in fp32. Slot k is scaled by
  ``topk_weights[:, k]``, the slots are summed in fp32 and the partial rounds to bf16 once.
- Masked routes (ids < 0 or >= E) contribute nothing.
- The kernel quantizes in other places (FP8 / bf16 exchange, its own MXFP4 rounding of the intermediate), so the
  tests compare the summed output by relative L2 over every rank: ``rel_l2 <= RTOL`` (0.1); graph replay and eager
  are bitwise equal, as are the ranks' outputs in the ar mode.

The fused tail (``tail_ref``): res_out = bf16(y + res_in), (Gemma)RMSNorm of res_out in fp32, FP8 E4M3 rows with one
fp32 scale per row ``amax(bf16(normed)) / 448``.

SpRsNorm (``sp_rs_norm_ref``): the fp32 sum of every rank's partial + the residual, Gemma RMSNorm; its router
(``sigmoid_topk_ref``): bf16 logits, sigmoid scores, top-k of score + bias, weights = scores / their sum * scale, then
the shared expert E with weight shared_w. Tests: rel L2 < 0.01 for the norm, >= 90% identical expert sets (bf16 logits
near ties) and weights within 2% on those rows.
"""

import torch
import torch.nn.functional as F

RTOL = 0.10

_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# |x| -> E2M1 code, round to nearest even: (upper bound, inclusive?) of codes 0..6, code 7 above
_E2M1_BOUNDS = ((0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True))


def _e8m0_roundup(amax: torch.Tensor) -> torch.Tensor:
    """uint8 biased exponent of ceil_pow2(amax / 6)."""
    u = (amax.float() / 6.0).view(torch.int32)
    e = (u >> 23) & 0xFF
    bump = ((u & 0x7FFFFF) != 0) & (e < 0xFF)
    return torch.where(bump, e + 1, e).to(torch.uint8)


def e8m0_to_f32(s: torch.Tensor) -> torch.Tensor:
    s = s.view(torch.uint8)
    bits = s.to(torch.int32) << 23
    bits = torch.where(s == 0, torch.full_like(bits, 0x00400000), bits)
    return bits.view(torch.float32)


def _e2m1_codes(y: torch.Tensor) -> torch.Tensor:
    a = y.abs()
    code = torch.zeros_like(a, dtype=torch.uint8)
    for bound, inclusive in _E2M1_BOUNDS:
        code += (a > bound if inclusive else a >= bound).to(torch.uint8)
    return code | (torch.signbit(y).to(torch.uint8) << 3)


def mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [..., K] -> (codes uint8 [..., K / 2], e8m0 scales uint8 [..., K / 32])."""
    shape = x.shape
    blocks = x.float().reshape(-1, 32)
    scale = _e8m0_roundup(blocks.abs().amax(-1))
    code = _e2m1_codes(blocks / e8m0_to_f32(scale)[:, None]).view(-1, 2)
    packed = code[:, 0] | (code[:, 1] << 4)
    return packed.view(*shape[:-1], shape[-1] // 2), scale.view(*shape[:-1], shape[-1] // 32)


def mxfp4_dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """(codes [..., K / 2], scales [..., K / 32]) -> fp32 [..., K]."""
    codes = codes.view(torch.uint8)
    lut = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float32, device=codes.device)
    vals = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], -1).flatten(-2)
    k = vals.shape[-1]
    vals = vals.view(*vals.shape[:-1], k // 32, 32) * e8m0_to_f32(scales.reshape(*vals.shape[:-1], k // 32))[..., None]
    return vals.view(*vals.shape[:-2], k)


def mxfp4_qdq(x: torch.Tensor) -> torch.Tensor:
    return mxfp4_dequant(*mxfp4_quant(x))


def _swiglu(g, u, limit=None):
    limit = 7.0 if limit is None else limit
    g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(1.702 * g) * (u + 1)


def _silu(g, u, limit=None):
    if limit:
        g, u = g.clamp(max=limit), u.clamp(min=-limit, max=limit)
    return F.silu(g) * u


def _situv2(g, u, beta=4.0, linear_beta=25.0):
    return beta * torch.tanh(g / beta) * torch.sigmoid(g) * linear_beta * torch.tanh(u / linear_beta)


ACTIVATIONS = ("silu", "swiglu", "situv2")


@torch.no_grad()
def mega_moe_tp_ref(x, topk_weights, topk_ids, w1, w1_scale, w2, w2_scale, activation="silu", *, beta=4.0,
                    linear_beta=25.0, swiglu_limit=None) -> torch.Tensor:
    """One rank's partial (bf16 values, fp32 tensor [M, H]) of the MoE of x [M, H] bf16 over this rank's inter slice:
    w1 / w2 the row-major codes [E, 2I, H / 2] / [E, H, I / 2] with scales [E, 2I, H / 32] / [E, H, I / 32]."""
    M, H = x.shape
    E, I2 = w1.shape[0], w1.shape[1]
    inter = I2 // 2
    ids = topk_ids.long()
    valid = (ids >= 0) & (ids < E)
    a1 = mxfp4_qdq(x)
    out = torch.zeros((M, topk_ids.shape[1], H), dtype=torch.float32, device=x.device)
    for e in torch.unique(ids[valid]).tolist():
        mask = ids == e
        rows = mask.nonzero()[:, 0]
        h = a1[rows] @ mxfp4_dequant(w1[e], w1_scale[e].view(I2, -1)).t()
        g, u = h[:, :inter], h[:, inter:]
        if activation == "swiglu":
            act = _swiglu(g, u, swiglu_limit)
        elif activation == "situv2":
            act = _situv2(g, u, beta, linear_beta)
        else:
            act = _silu(g, u, swiglu_limit)
        a2 = mxfp4_qdq(act.to(torch.bfloat16))
        out[mask] = a2 @ mxfp4_dequant(w2[e], w2_scale[e].view(H, -1)).t()
    out.mul_(torch.where(valid, topk_weights.float(), 0.0)[..., None])
    return out.sum(1).to(torch.bfloat16).float()


def tail_ref(y, res, norm_w, eps, gemma=True):
    """(res_out bf16, the fp32 normed rows, their per-row FP8 scales) of the fused tail."""
    r = y.float() + res.float()
    w = norm_w.float() + 1.0 if gemma else norm_w.float()
    xn = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * w
    scale = xn.to(torch.bfloat16).float().abs().amax(-1).clamp_min(1e-10) / 448.0
    return r.to(torch.bfloat16), xn, scale


def sp_rs_norm_ref(part_sum, res, norm_w, eps):
    """(res_out, out) fp32 of SpRsNorm's rows: part_sum the fp32 sum of the ranks' partials of those rows."""
    h = part_sum + res.float()
    return h, h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * (norm_w.float() + 1)


def sigmoid_topk_ref(normed_bf16, wg, bias, topk, scale):
    """(weights fp32 [M, topk], ids int32 [M, topk]) of SpRsNorm's router."""
    logits = (normed_bf16.float() @ wg.float().t()).to(torch.bfloat16).float()
    score = torch.sigmoid(logits)
    ids = (score + bias.float()).topk(topk, dim=-1).indices
    w = score.gather(1, ids)
    return w / w.sum(-1, keepdim=True) * scale, ids.to(torch.int32)
