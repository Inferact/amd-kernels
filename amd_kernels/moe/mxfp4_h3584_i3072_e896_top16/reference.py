# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference of the routed experts of one Kimi-K3.1 TP8 rank, the numerics aiter's a4w4 SiTUv2 ``fused_moe`` computes.

Contract (checkpoint-layout inputs, no preshuffle): ``x`` bf16 [M, 3584]; ``w13`` u8 [E, 768, 1792] (e2m1 pairs, low
nibble first; rows = 384 gate then 384 up), ``s13`` u8 [E, 768, 112] (e8m0 per 32); ``w2`` u8 [E, 3584, 192], ``s2``
u8 [E, 3584, 12]; ``topk_ids`` int [M, 16], ``topk_weights`` fp32 [M, 16].
  1. activation quant: MXFP4 per 32 along K; e8m0 = ceil exponent of fp32(amax * fp32(1/6)) (saturating at 254, floored
     at 92 for the activations),
     codes RNE e2m1 of x / 2^(e - 127) (saturating at 6);
  2. gate/up = fp32 sum of dequantized products; act = beta tanh(g/beta) sigmoid(g) * (lb tanh(u/lb)), beta 4, lb 25, no clamp;
  3. the act is quantized the same way along I (per 32 columns) and multiplied by w2 in fp32;
  4. slot k = topk_weights[:, k] * expert output, summed over the 16 slots in fp32, rounded once to bf16.
Tolerance used by the tests: kernel vs this reference, relative L2 <= 2e-2 end to end (fp4 codes flip on exact ties and on
fp32 sum order); two kernel runs are bit identical (no atomics).
"""
import torch

H, I, E, TOPK = 3584, 384, 896, 16  # noqa: E741
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_fp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """u8 [..., K/2] + e8m0 u8 [..., K/32] -> fp32 [..., K]."""
    lut = torch.cat([_E2M1, -_E2M1]).to(packed.device)
    lo, hi = packed & 0xF, packed >> 4
    v = torch.stack([lut[lo.long()], lut[hi.long()]], dim=-1).flatten(-2)
    sc = torch.pow(2.0, scales.float() - 127.0).repeat_interleave(32, dim=-1)
    return v * sc


def quant_mxfp4(x: torch.Tensor, floor: int = 0):
    """fp32/bf16 [..., K] -> (u8 codes [..., K/2], e8m0 u8 [..., K/32]) by the aiter rule; ``floor`` 92 for the
    activation quant (aiter's HIP kernel), 0 for the intermediate (GEMM1's epilogue)."""
    x = x.float()
    g = x.reshape(*x.shape[:-1], -1, 32)
    amax = g.abs().amax(-1)
    wi = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    bexp = ((wi + 0x7FFFFF) >> 23) & 0xFF
    bexp = torch.where(bexp.to(torch.int64) < 254, bexp, torch.full_like(bexp, 254)).clamp(min=floor)
    q = g / torch.pow(2.0, bexp.float() - 127.0).unsqueeze(-1)
    mag = q.abs().clamp(max=6.0)
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    idx = torch.bucketize(mag, mid, right=False)  # ties (exactly on a midpoint) go to the lower code
    tie_up = torch.isin(mag, mid) & (idx % 2 == 1)  # round half to even code index
    idx = torch.where(tie_up, idx + 1, idx)
    code = idx.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
    code = code.reshape(*x.shape[:-1], -1, 2)
    return (code[..., 0] | (code[..., 1] << 4)).contiguous(), bexp.to(torch.uint8)


def _tanh_act(g, u):
    b, lb = SITU_BETA, SITU_LINEAR_BETA
    return b * torch.tanh(g / b) * torch.sigmoid(g) * (lb * torch.tanh(u / lb))


def moe_ref(x, topk_ids, topk_weights, w13, s13, w2, s2):
    """Routed experts end to end; returns (out bf16 [M, 3584], slots fp32 [M, 16, 3584])."""
    m = x.shape[0]
    xq, xs = quant_mxfp4(x.float(), floor=92)
    xd = dequant_fp4(xq, xs)
    slots = torch.zeros(m, TOPK, H, dtype=torch.float32, device=x.device)
    for e in topk_ids.unique().tolist():
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        wg = dequant_fp4(w13[e], s13[e])
        gu = xd[t].double() @ wg.double().T
        act = _tanh_act(gu[:, :I].float(), gu[:, I:].float())
        aq, as_ = quant_mxfp4(act)
        w2d = dequant_fp4(w2[e], s2[e])
        y = dequant_fp4(aq, as_).double() @ w2d.double().T
        slots[t, k] = y.float() * topk_weights[t, k].unsqueeze(-1)
    return slots.sum(1).to(torch.bfloat16), slots
