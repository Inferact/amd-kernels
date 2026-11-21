# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``markov_gumbel_fused``: the Markov loop of vLLM's ``DSparkSpeculator._sample_sequential`` for a
Kimi-K3.1 draft (vocab 163840, Markov rank 512, ``draft_sample_method=probabilistic``, fp32 noise), from the base
logits of every draft position on.

For step s = 0..k-1 and request r (``prev`` = the anchor token at s = 0, else the step s-1 draw):
1. ``bias = bf16(sum_j f32(w1[prev, j]) * f32(w2[v, j]))``: fp32 accumulation, one bf16 rounding (the GEMV output).
2. ``y = bf16(f32(base[r, s, v]) + f32(bias[v]))``: one RNE rounding (a bf16 tensor add); stored into the draft logits
   cache row ``row(slot)`` column s (before temperature) when the row is >= 0.
3. ``T = temperature[slot]`` (0 for slot < 0). T = 0: ``u = y``; else ``u = y / T`` (IEEE fp32 division, skipped at
   T = 1) ``+ noise(v)`` with noise = ``-log(-log1p_neg_stable(p))`` and p = murmur3 uniform of
   (seed ^ (s << 32), sample_pos - 1 + 2^30, v) (``gumbel.py`` of vLLM, drafting salt).
4. ``draft_tokens[r, s] = argmax_v u`` (first index on ties; inputs are finite).

The sum order of step 1 differs from hipBLASLt / wvSplitK (any GEMM tiling does), so a bias value can differ in its
last bf16 bit; everything after it is exact. torch has no gfx950 ``v_log_f32``: the tests compare the fp64 noise within
2 fp32 ulps (``noise_approx``) and the sampled ids wherever the top two noised values differ by more than 1e-4.
"""

import torch

V, RANK = 163840, 512
M32 = 0xFFFFFFFF


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & M32


def _mix(h, key):
    key = (key * 0xCC9E2D51) & M32
    key = _rotl(key, 15)
    key = (key * 0x1B873593) & M32
    h = h ^ key
    h = _rotl(h, 13)
    return (h * 5 + 0xE6546B64) & M32


def _fmix(h):
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & M32
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & M32
    return h ^ (h >> 16)


def uniform_reference(seed: int, step: int, pos: int, cols: torch.Tensor) -> torch.Tensor:
    """murmur3_uniform32 of (seed ^ (step << 32), pos + 2^30, cols) as fp32."""
    s = (seed ^ (step << 32)) & ((1 << 64) - 1)
    h = _mix(_mix(_mix(0, s & M32), (s >> 32) & M32), (pos + (1 << 30)) & M32)
    c = cols.to(torch.int64)
    r = _fmix(_mix(torch.full_like(c, h), c) ^ 16)
    hi = (r >> 16).to(torch.float64)
    lo = (r & 0xFFFF).to(torch.float64)
    return (hi * 2.0**-16 + (lo + 0.5) * 2.0**-32).to(torch.float32)


def noise_approx(u: torch.Tensor) -> torch.Tensor:
    """-log(-log1p_neg_stable(u)) in fp64 from the fp32 u."""
    u = u.to(torch.float64)
    p = torch.full_like(u, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + u * p
    series = -u * p
    direct = torch.log(torch.clamp(1.0 - u, min=5.960464477539063e-08))
    l1 = torch.where(u < 0.25, series, direct)
    return (-torch.log(-l1)).to(torch.float32)


def bias_reference(w1, w2, prev):
    """bf16 [n, V]: bf16(fp32 accumulation of w1[prev] . w2^T)."""
    return (w1[prev].float() @ w2.float().t()).to(torch.bfloat16)


def logits_reference(base, bias):
    """bf16 [n, V]: bf16(f32(base) + f32(bias))."""
    return (base.float() + bias.float()).to(torch.bfloat16)


def noised_reference(y, temp, seed, step, pos):
    """fp32 [V] noised values of one row (``y`` bf16 [V])."""
    yf = y.float()
    if temp == 0.0:
        return yf
    v = yf if temp == 1.0 else yf / temp
    u = uniform_reference(seed, step, pos, torch.arange(V, device=y.device))
    return v + noise_approx(u)


def markov_gumbel_reference(base, w1, w2, anchor, temp, seed, sample_pos):
    """Reference chain: ``base`` bf16 [n, k, V]; ``anchor`` int64 [n] (the step 0 previous tokens); per-row ``temp``
    fp32 [n], ``seed`` int64 [n] and ``sample_pos`` int64 [n, k] (the draw is at ``sample_pos - 1``). Returns
    (tokens int64 [n, k], logits bf16 [n, k, V], noised fp32 [n, k, V])."""
    n, k, _ = base.shape
    toks = torch.zeros(n, k, dtype=torch.int64, device=base.device)
    logits = torch.zeros(n, k, V, dtype=torch.bfloat16, device=base.device)
    noised = torch.zeros(n, k, V, dtype=torch.float32, device=base.device)
    prev = anchor.clone()
    for s in range(k):
        y = logits_reference(base[:, s], bias_reference(w1, w2, prev))
        logits[:, s] = y
        for r in range(n):
            noised[r, s] = noised_reference(y[r], float(temp[r]), int(seed[r]), s, int(sample_pos[r, s]) - 1)
        prev = noised[:, s].argmax(-1)
        toks[:, s] = prev
    return toks, logits, noised
