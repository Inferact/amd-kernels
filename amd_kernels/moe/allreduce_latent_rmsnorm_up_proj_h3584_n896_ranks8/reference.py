# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract of the eight-rank latent tail of Kimi-K3.1 decode: all-reduce, RMSNorm(3584), up projection (hidden 7168 / 8).

Input per rank: the routed-expert latent partial ``parts[r]`` (bf16 [rows, 3584]), the RMSNorm weight (bf16 [3584]),
this rank's up-projection shard ``w`` (bf16 [896, 3584], rows [896 rank, 896 (rank + 1)) of the [7168, 3584] weight)
and ``shared`` (bf16 [rows, 896], the shared-expert output of this rank's hidden shard: the TP partial in the stock
layout, the full output in the replicated layout). Output: ``out`` (bf16 [rows, 896]).

Reduction: the eight partials are added in rank order 0..7 in fp32 and rounded once to bf16 (the latent). Norm:
``xn = x * rsqrt(mean(x * x) + eps) * w_norm`` of the bf16 latent (plain RMSNorm, no weight offset). Up projection:
``out = bf16(shared + xn @ w^T)`` with the sum in fp32. The kernel evaluates ``bf16(x * w_norm) @ w^T`` in fp32 and
applies ``rsqrt(mean(x * x) + eps)`` to the row sum afterwards (the stock fused Triton kernel's order), so its error
against the fp64 evaluation of this function is the bf16 rounding of ``x * w_norm`` plus the final rounding.

Tolerance of the tests: the latent bit for bit; the output within 2^-8 |y| + 2^-8 max|xn @ w^T| of the fp64
evaluation ``y`` of the same bf16 inputs, cosine >= 0.99999 per row; two runs match bit for bit.
"""
import torch


def reduce_reference(parts):
    """bf16 of the fp32 sum of the partials in rank order."""
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.bfloat16()


def up_proj_reference(latent, norm_weight, w, shared, eps=1e-6):
    """(y fp64 [rows, 896] = shared + rmsnorm(latent) @ w^T, up fp64 [rows, 896] = the product alone)."""
    x = latent.double()
    xn = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * norm_weight.double()
    up = xn @ w.double().T
    return shared.double() + up, up


def latent_tail_reference(parts, norm_weight, w, shared, eps=1e-6):
    """(y, up, latent) for the eight partials ``parts``."""
    latent = reduce_reference(parts)
    y, up = up_proj_reference(latent, norm_weight, w, shared, eps)
    return y, up, latent


def slot_sum_reference(slots, reverse_sorted):
    """The latent partial [rows, 3584] of one rank from the routed experts' slot buffer ([sorted rows, 3584] bf16, already
    multiplied by the router weights): token t's slot k is row ``reverse_sorted[16 t + k]``; the 16 slots are added in
    order k = 0..15 in fp32 and rounded once to bf16 (the same bits as the routed-experts kernel's own sum)."""
    rows = reverse_sorted.view(-1, 16).long()
    acc = slots[rows[:, 0]].float()
    for k in range(1, 16):
        acc = acc + slots[rows[:, k]].float()
    return acc.bfloat16()
