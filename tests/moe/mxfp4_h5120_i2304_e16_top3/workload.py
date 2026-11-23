# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared test and benchmark inputs: MXFP4 experts, MXFP8 activations and routing of the served shape."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e16_top3.reference import E, H, I, TOPK, quantize_mxfp8

# draft rows per rank: 5 per request (B 3-205); routed rows per rank = 3 x tokens (8 ranks x T x 3 / 8)
TOKENS = (16, 40, 80, 160, 320, 640, 1024)


def require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"gfx950 only, found {arch}")


def make_experts(seed: int = 0):
    """Random MXFP4 experts, scales over 6 binades: (w13 codes ``[16, 4608, 2560]``, w13 e8m0 ``[16, 4608, 160]``,
    w2 codes ``[16, 5120, 1152]``, w2 e8m0 ``[16, 5120, 72]``), row-major, uint8."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g)

    return (rnd(0, 256, (E, 2 * I, H // 2)), rnd(118, 124, (E, 2 * I, H // 32)),
            rnd(0, 256, (E, H, I // 2)), rnd(118, 124, (E, H, I // 32)))


def make_activation(n: int, seed: int = 0, scale: float = 1.0):
    """MXFP8 activations: (e4m3 codes ``[n, 5120]``, e8m0 ``[n, 160]``)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return quantize_mxfp8((torch.randn((n, H), device="cuda", generator=g) * scale).bfloat16())


def make_routes(n: int, dist: str = "uniform", seed: int = 0):
    """(topk_ids i32 ``[n, 3]``, topk_weights f32 ``[n, 6]``): 3 distinct experts per token. ``uniform``: equal odds;
    ``skew``: odds ~ 1 / (rank + 1) over a per-seed random expert order (the hottest expert gets ~1 token in 3)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.zeros(E, device="cuda")
    if dist == "skew":
        order = torch.randperm(E, device="cuda", generator=g)
        logits[order] = -torch.log(torch.arange(1, E + 1, device="cuda").float())
    gumbel = -torch.log(-torch.log(torch.rand((n, E), device="cuda", generator=g).clamp_min(1e-20)))
    ids = (logits + gumbel).topk(TOPK, dim=1).indices.int()
    return ids, torch.softmax(torch.randn((n, TOPK), device="cuda", generator=g), -1)
