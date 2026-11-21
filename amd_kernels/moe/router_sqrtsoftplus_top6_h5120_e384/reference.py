# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The torch contract of ``router_topk``: the served gate GEMM and top-k with the sqrt-softplus score.

    logits = x @ weight.T                              fp32 accumulate of the bf16 products
    score  = sqrt(logit if logit > 20 else log(1 + exp(logit)))
    ids    = the 6 experts with the largest score + bias, descending (ties: the lowest expert id)
    w      = ((score + bias)[ids] - bias[ids]) * (routed_scaling_factor / sum of those, in slot order)
    padding rows: ids -1, weights 0

The weight of a pick is ``(score + bias) - bias`` in fp32, not ``score``: vLLM's kernel decodes it from the argmax key.
Tolerance: the kernel computes exp with the fast exp2, so its scores differ from torch's by an ulp; its ids and weights
equal the served ``topk_hash_softplus_sqrt`` of the same logits bit for bit (tests), and the reference's ids except
where two ``score + bias`` values within a few fp32 ulp compete for the 6th place (the tests give the margin), its
weights within 2e-6 relative.
"""

import torch


def logits_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, weight.T, out_dtype=torch.float32)


def score_reference(logits: torch.Tensor) -> torch.Tensor:
    return torch.where(logits > 20.0, logits, torch.log(1.0 + torch.exp(logits))).sqrt()


def topk_reference(logits, bias, routed_scaling_factor=1.5, is_padding=None):
    score = score_reference(logits.float())
    ids = torch.topk(score + bias, 6, dim=-1)[1]
    w = (score + bias).gather(1, ids) - bias[ids]
    total = w[:, 0]
    for k in range(1, 6):
        total = total + w[:, k]
    w = w * (routed_scaling_factor / total[:, None])
    ids = ids.int()
    if is_padding is not None:
        w = torch.where(is_padding[:, None], 0.0, w)
        ids = torch.where(is_padding[:, None], -1, ids)
    return w, ids


def router_topk_reference(x, weight, bias, routed_scaling_factor=1.5, is_padding=None):
    return topk_reference(logits_reference(x, weight), bias, routed_scaling_factor, is_padding)
