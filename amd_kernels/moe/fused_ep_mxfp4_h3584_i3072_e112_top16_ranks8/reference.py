# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference for MXFP8 inputs, SiTUv2 and fixed-order BF16 slot reduction."""
import torch

H, I, E, TOPK = 3584, 3072, 112, 16
_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Row-major MXFP4 (``[.., K/2]`` u8, ``[.., K/32]`` e8m0 u8) -> f32 ``[.., K]``."""
    lut = torch.cat([_GRID, -_GRID]).to(codes.device)
    v = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1).flatten(-2)
    return v * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)


def dequant_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """e4m3 ``[M, K]`` + e8m0 ``[M, K/32]`` -> f32 ``[M, K]``."""
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scales.float() - 127.0).unsqueeze(-1)).reshape(m, k)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """f32 / bf16 ``[M, K]`` -> e4m3 codes ``[M, K]``, e8m0 bytes ``[M, K/32]`` (the exponent of
    ``ceil_pow2(amax * fp32(1 / 448))``, at least 1)."""
    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    u = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    e = torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp_min(1)
    codes = (blocks * torch.exp2(127.0 - e.float()).unsqueeze(-1)).clamp(-448.0, 448.0).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), e.to(torch.uint8)


def ep_reference(codes, scales, weights, topk_weights, topk_ids):
    x = dequant_mxfp8(codes, scales)
    slots = x.new_zeros((*topk_ids.shape, H), dtype=torch.bfloat16)
    for owner, (w13, s13, w2, s2) in enumerate(weights):
        ids = topk_ids - owner * w13.shape[0]
        for expert in ids[(ids >= 0) & (ids < w13.shape[0])].unique().tolist():
            token, slot = (ids == expert).nonzero(as_tuple=True)
            gate, up = (x[token] @ dequant_mxfp4(w13[expert], s13[expert]).T).chunk(2, -1)
            act = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))
            q = dequant_mxfp8(*quantize_mxfp8(act))
            y = q @ dequant_mxfp4(w2[expert], s2[expert]).T
            slots[token, slot] = (y * topk_weights[token, slot, None]).bfloat16()
    acc = slots[:, 0].float()
    for slot in range(1, TOPK):
        acc = acc + slots[:, slot].float()
    return acc.bfloat16()
