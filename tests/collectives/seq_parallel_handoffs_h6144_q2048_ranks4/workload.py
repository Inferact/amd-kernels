# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared inputs of the hand-off tests: the served token counts, the weights (o_proj_full, norms, the MoE block with
its shared expert) and seeded per-rank inputs (every rank can rebuild every rank's)."""

import torch

H, Q, WORLD = 6144, 2048, 4
# the SP range's edges and the counts the vLLM recipe serves (8885 / 15745: single-batch prefills, 32768: the cap)
SERVED_TOKENS = (4096, 4097, 8885, 15745, 32768)


def rand(shape, seed, dev, scale=1.0, dtype=torch.bfloat16):
    g = torch.Generator(device=dev).manual_seed(seed)
    return (torch.randn(shape, device=dev, generator=g) * scale).to(dtype)


def o_proj_weight(dev):
    """A preshuffled MXFP8 o_proj_full (N 6144, K 8192): codes and their scales."""
    from amd_kernels.gemm.mxfp8_common.gemm import preshuffle_mxfp8_weight

    g = torch.Generator(device=dev).manual_seed(5)
    wq = torch.randint(0, 120, (H, WORLD * Q), generator=g, device=dev, dtype=torch.uint8)
    wq = torch.where(torch.rand(wq.shape, generator=g, device=dev) < 0.5, wq, wq | 0x80)  # both signs
    wsc = torch.randint(116, 121, (H, WORLD * Q // 32), generator=g, device=dev, dtype=torch.uint8)
    return preshuffle_mxfp8_weight(wq.view(torch.float8_e4m3fn), wsc)


def moe_weights(dev):
    """The MoE block (router, biased top-4, MXFP4 routed experts, the MXFP8 shared expert) as MoeWeights."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import MoeWeights, SharedWeights
    from tests.moe.mxfp4_h6144_i768_e128_top4.workload import make_experts

    raw, _ = make_experts()
    g = torch.Generator(device=dev).manual_seed(607)

    def weight(n, k):
        q = (torch.randn((n, k), device=dev, generator=g) * 20).to(torch.float8_e4m3fn).view(torch.uint8)
        return q, torch.randint(118, 124, (n, k // 32), device=dev, dtype=torch.uint8, generator=g)

    shared = SharedWeights.from_row_major(*weight(1536, H), *weight(H, 768))
    g2 = torch.Generator(device=dev).manual_seed(5)
    router = torch.randn(128, H, device=dev, generator=g2) / H**0.5
    bias = torch.randn(128, device=dev, generator=g2) * 1e-2
    return MoeWeights.from_row_major(router, bias, *raw, shared=shared)
