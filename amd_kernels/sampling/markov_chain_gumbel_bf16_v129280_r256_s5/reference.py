# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``markov_chain``: the sequential Markov draft sampling of a DSpark speculator, 5 steps, vocabulary
129280, Markov rank 256, any number of requests (``DSparkSpeculator._sample_sequential`` with
``draft_sample_method=probabilistic``, bf16 logits, fp32 Gumbel draws).

Row r (a request), step s = 0..4, previous token p (step 0: ``input_ids[anchor_idx[r]]``, then the draft token of step
s - 1):
1. ``bias = bf16(sum_k w1[p, k] w2[v, k])``, an fp32 sum of 256 products, one bf16 rounding (the order of the sum is
   the kernel's: the bias can differ from another order in its last bf16 bit).
2. ``y = bf16(f32(base[r, s]) + f32(bias))``, stored as ``cache[slot, s]`` (pre-temperature) when the slot
   ``idx_map[r, s]`` >= 0.
3. ``T = temperature[slot]`` (0 for slot < 0): T 0: ``v = y``, else ``v = y / T`` (fp32 division) + noise, noise =
   ``-log(-log1p_neg_stable(u))``, u = murmur3 uniform of (seed ^ (s << 32), ``sample_pos[r, s] - 1`` + 2^30, vocab id).
4. ``tokens[r, s]`` = the lowest vocab id of the maximum of v (NaN and +inf values never win: the kernel's residual
   y - T (y / T) is NaN for +inf). A row with no winner (every v NaN, +inf or -inf) yields token 0: a token is always in [0, V) and the next step's embedding row exists.
The torch functions below give each piece for given previous tokens, so a test follows the kernel's own chain
(teacher forcing): torch has no gfx950 ``v_log_f32``, the noise is fp64 and agrees with the kernel's within ~2 fp32
ulps, so ids are compared except where the top two values are closer than 1e-4 + 1e-6 |top|.
"""

import torch

V, R, NSPEC = 129280, 256, 5
TILE = 512  # vocabulary columns of a workgroup

_M32 = 0xFFFFFFFF


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & _M32


def _mix(h, key):
    key = (key * 0xCC9E2D51) & _M32
    key = _rotl(key, 15)
    key = (key * 0x1B873593) & _M32
    h = h ^ key
    h = _rotl(h, 13)
    return (h * 5 + 0xE6546B64) & _M32


def _tail(h, c):
    """murmur3_hash32 after the position, for the vocab ids ``c`` (int64 tensor) at the prefix hash ``h``."""
    key = (c * 0xCC9E2D51) & _M32
    key = ((key << 15) | (key >> 17)) & _M32
    key = (key * 0x1B873593) & _M32
    x = torch.full_like(c, h) ^ key
    x = ((x << 13) | (x >> 19)) & _M32
    x = (x * 5 + 0xE6546B64) & _M32
    x = x ^ 16
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & _M32
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & _M32
    return x ^ (x >> 16)


def uniform_reference(seed: int, step: int, pos: int, cols: torch.Tensor) -> torch.Tensor:
    """murmur3_uniform32 of (seed ^ (step << 32), pos + 2^30, cols) as fp32 (two exact halves, one rounding)."""
    s = (seed ^ (step << 32)) & ((1 << 64) - 1)
    h = _mix(_mix(_mix(0, s & _M32), (s >> 32) & _M32), (pos + (1 << 30)) & _M32)
    c = cols.to(torch.int64)
    r = _tail(h, c)
    hi = (r >> 16).to(torch.float64)
    lo = (r & 0xFFFF).to(torch.float64)
    return (hi * 2.0 ** -16 + (lo + 0.5) * 2.0 ** -32).to(torch.float32)


def noise_approx(u: torch.Tensor) -> torch.Tensor:
    """-log(-log1p_neg_stable(u)) in fp64 from the fp32 u (the kernel's is fp32: within ~2 ulps of this)."""
    u = u.to(torch.float64)
    p = torch.full_like(u, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + u * p
    series = -u * p
    direct = torch.log(torch.clamp(1.0 - u, min=5.960464477539063e-08))
    l1 = torch.where(u < 0.25, series, direct)
    return (-torch.log(-l1)).to(torch.float32)


def bias_reference(prev: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """bf16 [n, V]: bf16(w1[prev] @ w2^T) with fp32 sums."""
    return (w1[prev].float() @ w2.float().t()).to(torch.bfloat16)


def logits_reference(base: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """bf16 [n, V]: bf16(f32(base) + f32(bias)), the draft-cache row."""
    return (base.float() + bias.float()).to(torch.bfloat16)


def sample_reference(y: torch.Tensor, temp: torch.Tensor, seed: torch.Tensor, step: int, pos: torch.Tensor):
    """(ids int64 [n], gap fp32 [n]): the argmax of the noised values of the cache rows ``y`` bf16 [n, V]; ``temp``
    fp32 [n], ``seed`` / ``pos`` int64 [n] (pos = sample_pos - 1) of each row; gap = top - second value."""
    n = y.shape[0]
    cols = torch.arange(V, device=y.device)
    vals = torch.empty(n, V, dtype=torch.float32, device=y.device)
    for r in range(n):
        yr = y[r].float()
        if float(temp[r]) == 0.0:
            vals[r] = yr
        else:
            u = uniform_reference(int(seed[r]), step, int(pos[r]), cols)
            vals[r] = yr / temp[r] + noise_approx(u)
    top = vals.topk(2, dim=1)
    vals = torch.where(vals.isnan() | (vals == float("inf")), torch.full_like(vals, float("-inf")), vals)
    return vals.argmax(dim=1), top.values[:, 0] - top.values[:, 1]


def markov_chain_reference(base, w1, w2, input_ids, anchor_idx, idx_map, sample_pos, temperature, seeds, cache,
                           tokens) -> None:
    """The whole chain in torch with the entry's arguments: writes the cache rows (slot >= 0) and ``tokens`` [n, 5].
    The noise is the fp64 one (~2 ulps from the kernel's), the bias sums are torch's: tokens follow the kernel's
    except near ties and where a last-bit bias difference changes a row's chain."""
    n = anchor_idx.shape[0]
    prev = input_ids[anchor_idx].long()
    b3 = base.view(n, NSPEC, V)
    slots = idx_map.long()
    for s in range(NSPEC):
        y = logits_reference(b3[:, s], bias_reference(prev, w1, w2))
        valid = slots[:, s] >= 0
        slot = slots[:, s].clamp(min=0)
        cache[slot[valid], s] = y[valid]
        temp = torch.where(valid, temperature[slot], torch.zeros_like(temperature[slot]))
        tokens[:n, s], _ = sample_reference(y, temp, seeds[slot], s, sample_pos[:, s] - 1)
        prev = tokens[:n, s]
