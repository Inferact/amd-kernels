# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared test inputs of the fused expert-parallel MoE: per-rank MXFP4 experts, MXFP8 / bf16 tokens, global routes."""

import torch
from amd_kernels.moe.fused_ep_mxfp4_h5120_i2304_e48_top6_ranks8.reference import H, I, TOPK, quantize_mxfp8


def make_experts(seed: int, device, epr: int = 48):
    """One rank's random MXFP4 experts, row-major, uint8: (w13 ``[epr, 4608, 2560]``, w13 e8m0 ``[epr, 4608, 160]``,
    w2 ``[epr, 5120, 1152]``, w2 e8m0 ``[epr, 5120, 72]``)."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device=device, generator=g)

    return (rnd(0, 256, (epr, 2 * I, H // 2)), rnd(118, 124, (epr, 2 * I, H // 32)), rnd(0, 256, (epr, H, I // 2)),
            rnd(118, 124, (epr, H, I // 32)))


def make_routes(n: int, world: int, kind: str, seed: int, device, epr: int = 48):
    """(ids i32 ``[n, 6]`` global expert ids, weights f32 ``[n, 6]``): 6 distinct experts per token. ``uniform``:
    equal odds; ``skew``: odds ~ 1 / (rank + 1) over a fixed random order; ``hot``: every route on rank 0."""
    g = torch.Generator(device=device).manual_seed(seed)
    ne = epr * world
    logits = torch.zeros(ne, device=device)
    if kind == "skew":
        order = torch.randperm(ne, device=device, generator=torch.Generator(device=device).manual_seed(99))
        logits[order] = -torch.log(torch.arange(1, ne + 1, device=device).float())
    elif kind == "hot":
        logits[epr:] = -1e4
    gumbel = -torch.log(-torch.log(torch.rand((n, ne), device=device, generator=g).clamp_min(1e-20)))
    ids = (logits + gumbel).topk(TOPK, dim=1).indices.int()
    return ids, torch.softmax(torch.randn((n, TOPK), device=device, generator=g), -1)


def make_tokens(n: int, seed: int, device):
    """bf16 rows and their MXFP8 form (e4m3 codes, e8m0 ``[n, 160]``)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((n, H), device=device, generator=g).bfloat16()
    codes, scales = quantize_mxfp8(x)
    return x, codes, scales
