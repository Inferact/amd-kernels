# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared test and benchmark inputs: MXFP4 experts, MXFP8 activations and routing of the served shape."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h5120_i2304_e48_top6.reference import E, H, I, TOPK, quantize_mxfp8

# routed rows per rank = 6 x tokens (8 ranks x T x 6 / 8)
TOKENS = (16, 32, 64, 128, 192, 384, 768)


def require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"gfx950 only, found {arch}")


def make_experts(seed: int = 0, e: int = E):
    """Random MXFP4 experts, scales over 6 binades: (w13 codes ``[e, 4608, 2560]``, w13 e8m0 ``[e, 4608, 160]``,
    w2 codes ``[e, 5120, 1152]``, w2 e8m0 ``[e, 5120, 72]``), row-major, uint8."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(lo, hi, shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g)

    return (rnd(0, 256, (e, 2 * I, H // 2)), rnd(118, 124, (e, 2 * I, H // 32)),
            rnd(0, 256, (e, H, I // 2)), rnd(118, 124, (e, H, I // 32)))


def make_activation(n: int, seed: int = 0, scale: float = 1.0):
    """MXFP8 activations: (e4m3 codes ``[n, 5120]``, e8m0 ``[n, 160]``)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    return quantize_mxfp8((torch.randn((n, H), device="cuda", generator=g) * scale).bfloat16())


def make_routes(n: int, dist: str = "uniform", seed: int = 0, e: int = E):
    """(topk_ids i32 ``[n, 6]``, topk_weights f32 ``[n, 6]``): 6 distinct experts per token. ``uniform``: equal odds;
    ``skew``: odds ~ 1 / (rank + 1) over a per-seed random expert order (the hottest expert gets ~1 token in 3)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.zeros(e, device="cuda")
    if dist == "skew":
        order = torch.randperm(e, device="cuda", generator=g)
        logits[order] = -torch.log(torch.arange(1, e + 1, device="cuda").float())
    gumbel = -torch.log(-torch.log(torch.rand((n, e), device="cuda", generator=g).clamp_min(1e-20)))
    ids = (logits + gumbel).topk(TOPK, dim=1).indices.int()
    return ids, torch.softmax(torch.randn((n, TOPK), device="cuda", generator=g), -1)


def make_sorted_rows(n_rows: int, dist: str = "uniform", seed: int = 0, e: int = E):
    """Receive-buffer rows already sorted by expert: (``expert_offsets`` i32 ``[49]``, ``row_weights`` f32
    ``[n_rows]``), the experts' row counts drawn from ``make_routes`` over ``n_rows / 6`` tokens (``n_rows`` a multiple
    of 6)."""
    ids, _ = make_routes(n_rows // TOPK, dist, seed, e)
    counts = torch.bincount(ids.flatten().long(), minlength=e)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device="cuda"), counts.cumsum(0)]).int()
    g = torch.Generator(device="cuda").manual_seed(seed + 77)
    return offsets, torch.rand(n_rows, device="cuda", generator=g) + 0.1
