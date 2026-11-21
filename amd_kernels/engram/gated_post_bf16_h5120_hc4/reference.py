# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``gated_post``: vLLM's ``_fused_engram_post_wkv_kernel`` (``common/engram.py``).

hidden bf16 [T, 4, 5120], kv bf16 [T, 25600] (the ``wkv`` output: four keys of 5120, then the value), q_weight and
k_weight bf16 [4, 5120], optional token_mask bool [T]. Per (token, hc), in fp32: ``hidden_rms = rsqrt(mean(h^2) +
eps)``, ``key_rms = rsqrt(mean(key^2) + eps)``, ``dot = sum(h * q * k * key) * hidden_rms * key_rms / sqrt(5120)``,
``gate = sigmoid(sign(dot) * sqrt(max(|dot|, clamp)))`` (0 where the mask is False), output ``hidden + gate * value``
rounded once to bf16. The reference runs in fp32 with torch's reductions; the kernel's sums differ in association, so
tests compare with cosine >= 0.99999 and a bounded bf16 difference (at most one bf16 ulp of the output).
"""

import torch

H = 5120
HC = 4
EPS = 1e-20
CLAMP = 1e-6


def gated_post_reference(hidden, kv, q_weight, k_weight, token_mask=None, eps=EPS, clamp=CLAMP):
    t = hidden.shape[0]
    h = hidden.float()
    key = kv[:, : HC * H].float().view(t, HC, H)
    value = kv[:, HC * H :].float().view(t, 1, H)
    h_rms = torch.rsqrt(h.square().mean(-1) + eps)
    k_rms = torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * q_weight.float() * k_weight.float() * key).sum(-1) * h_rms * k_rms * (H**-0.5)
    mag = torch.sqrt(torch.clamp(dot.abs(), min=clamp))
    gate = torch.sigmoid(torch.where(dot < 0, -mag, mag))
    if token_mask is not None:
        gate = torch.where(token_mask.view(t, 1), gate, torch.zeros_like(gate))
    return (h + gate.unsqueeze(-1) * value).to(torch.bfloat16)
