# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served shapes, random checkpoint-layout weights and routing for the tests and benchmarks."""
import torch

from amd_kernels.moe.mxfp4_h3584_i3072_e896_top16.reference import E, H, I, TOPK

# critical M and the CUDA graph padding sizes up to 192
SERVED_M = [1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 65, 68, 80, 96, 128, 160, 192]


def make_checkpoint(seed=0, device="cuda"):
    """Checkpoint-layout expert weights: e2m1 pairs (random bytes) and e8m0 scales around 2^-7."""
    g = torch.Generator(device=device).manual_seed(seed)
    r = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, device=device, generator=g)
    sc = lambda *s: torch.randint(118, 124, s, dtype=torch.uint8, device=device, generator=g)
    return r(E, 2 * I, H // 2), sc(E, 2 * I, H // 32), r(E, H, I // 2), sc(E, H, I // 32)


def make_routing(m, dist="uniform", seed=0, device="cuda"):
    """topk_ids int32 [m, 16] (distinct experts per token) and fp32 softmax weights; ``zipf`` skews expert popularity."""
    g = torch.Generator(device=device).manual_seed(seed + 1000 * m)
    logit = torch.zeros(E, device=device)
    if dist == "zipf":
        logit = -torch.log(torch.arange(1, E + 1, device=device, dtype=torch.float32))
        logit = logit[torch.randperm(E, device=device, generator=g)]
    gum = -torch.log(-torch.log(torch.rand(m, E, device=device, generator=g).clamp_min(1e-9)))
    ids = (logit + gum).topk(TOPK, dim=1).indices.to(torch.int32)
    w = torch.softmax(torch.randn(m, TOPK, device=device, generator=g), 1)
    return ids.contiguous(), w.contiguous()


def make_activation(m, seed=0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed + 7 * m)
    return (torch.randn(m, H, device=device, generator=g) * 0.5).to(torch.bfloat16)
