# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``rejection_sample``: standard speculative rejection sampling (Leviathan et al.) of 5 draft tokens
per request over a vocabulary of 129280, with a bonus row (6 logits rows per request), as vLLM's ``rejection_sample``
runs it for ``rejection_sample_method=standard`` with a replicated draft-logits cache, no watermark, no block
verification and no FLy window.

Inputs: ``target_logits`` [B * 6, V]: fp32 with the temperature applied (and any top-k / top-p mask), or bf16 raw
logits, which the kernel divides by the request's temperature in fp32 (vLLM's ``apply_temperature``; temperature 0
compares the raw logits); ``draft_logits`` bf16
[R, 5, V] (pre-temperature, row = request state), ``draft_sampled`` int64 [B * 6] (row 6 b + j + 1 is the draft token
of step j, < 0: a placeholder that is always rejected), ``pos`` int64 [B * 6], ``idx_mapping`` int32 [B],
``temperature`` fp32 [R], ``seed`` int64 [R]; optional ``synthetic_rates`` fp32 [5] (conditional acceptance rates).

Per request, step j = 0..4 in order until the first rejection (``acc`` = the accepted count):
1. ``u = max(philox(seed, pos[6 b + j]), 2^-31)``: Triton's ``tl.rand`` (Philox 4x32-10, key (seed lo, seed hi),
   counter (pos lo, pos hi, 0, 0), ``int32 -> [0, 1)`` with scale 4.6566127342e-10).
2. Temperature 0: accept iff the draft token is the lowest-id argmax of the target row. Otherwise accept iff
   ``log p(x) > log u + log q(x)`` with ``log p = target - lse(target)``, ``log q = draft / T - lse(draft / T)``.
   ``synthetic_rates``: accept iff ``u < rates[j]`` instead.
3. Row ``acc`` of the request gets one resampled token: temperature 0: the argmax of the target row; otherwise the
   argmax of ``residual + gumbel`` where ``residual`` is the target row for the bonus row (acc 5) or a placeholder
   draft, else ``log p + log1p(-q / p)`` where ``q < p`` and -inf elsewhere (the lowest id on a tie, id 0 when
   every value is -inf); ``gumbel = -log(-log1p_neg_stable(u))`` for the murmur3 uniform of (seed, pos[6 b + acc],
   vocab id).
Output: ``sampled`` int64 [B, 6] (the draft tokens of the accepted steps, the resampled token at ``acc``, -1 after)
and ``num_sampled`` int32 [B] = ``acc + 1``.

The reference works in fp64 over the same fp32 / bf16 inputs; the kernel's fp32 sums, exp and log differ in the
last bits, so a decision (an acceptance test or an argmax) can differ only where its margin is tiny: ``margin``
fp32 [B] is the smallest slack of any decision of the request, and tests compare tokens where it exceeds 1e-3.
"""

import numpy as np
import torch

V = 129280
NSPEC = 5
ROWS = NSPEC + 1
U_MIN = 4.6566127342e-10
_M32 = 0xFFFFFFFF
_M64 = (1 << 64) - 1


def philox_uniform(seed: int, pos: int) -> float:
    """Triton's ``tl.rand(seed, pos)`` as an fp32 value, clamped to the smallest positive output."""
    k0, k1 = seed & _M32, (seed >> 32) & _M32
    c0, c1, c2, c3 = pos & _M32, (pos >> 32) & _M32, 0, 0
    for _ in range(10):
        a, b = c0, c2
        c0 = (((0xCD9E8D57 * b) >> 32) ^ c1 ^ k0) & _M32
        c2 = (((0xD2511F53 * a) >> 32) ^ c3 ^ k1) & _M32
        c1 = (0xCD9E8D57 * b) & _M32
        c3 = (0xD2511F53 * a) & _M32
        k0 = (k0 + 0x9E3779B9) & _M32
        k1 = (k1 + 0xBB67AE85) & _M32
    x = c0 - (1 << 32) if c0 >= (1 << 31) else c0
    x = -x - 1 if x < 0 else x
    return max(float(np.float32(x) * np.float32(U_MIN)), U_MIN)


def _rotl(x, s):
    return ((x << s) | (x >> (32 - s))) & _M32


def _mix(h, key):
    key = (key * 0xCC9E2D51) & _M32
    key = _rotl(key, 15)
    key = (key * 0x1B873593) & _M32
    return (_rotl(h ^ key, 13) * 5 + 0xE6546B64) & _M32


def gumbel_noise(seed: int, pos: int, cols: torch.Tensor) -> torch.Tensor:
    """fp64 ``-log(-log1p_neg_stable(u))`` of the fp32 murmur3 uniform of (seed, pos, column)."""
    h = _mix(_mix(_mix(0, seed & _M32), (seed >> 32) & _M32), pos & _M32)
    c = cols.to(torch.int64)
    key = (c * 0xCC9E2D51) & _M32
    key = ((key << 15) | (key >> 17)) & _M32
    key = (key * 0x1B873593) & _M32
    x = torch.full_like(c, h) ^ key
    x = ((x << 13) | (x >> 19)) & _M32
    x = ((x * 5 + 0xE6546B64) & _M32) ^ 16
    x = x ^ (x >> 16)
    x = (x * 0x85EBCA6B) & _M32
    x = x ^ (x >> 13)
    x = (x * 0xC2B2AE35) & _M32
    r = x ^ (x >> 16)
    u = ((r >> 16).double() * 2.0**-16 + ((r & 0xFFFF).double() + 0.5) * 2.0**-32).float().double()
    p = torch.full_like(u, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + u * p
    l1 = torch.where(u < 0.25, -u * p, torch.log(torch.clamp(1.0 - u, min=5.960464477539063e-08)))
    return -torch.log(-l1)


def _log1p_neg_stable(v: torch.Tensor) -> torch.Tensor:
    p = torch.full_like(v, 1.0 / 8.0)
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = 1.0 / k + v * p
    return torch.where(v < 0.25, -v * p, torch.log(torch.clamp(1.0 - v, min=5.960464477539063e-08)))


def _argmax_lowest(x: torch.Tensor) -> int:
    """The lowest index of the maximum (torch.argmax does not promise it)."""
    return int((x == x.max()).nonzero()[0])


def _top_gap(x: torch.Tensor) -> float:
    t = x.topk(2).values
    return float(t[0] - t[1])


def _target_row(target_logits, row, temp):
    """fp64 target row: an fp32 row as is, a bf16 row divided by the temperature in fp32 (not at temperature 0)."""
    x = target_logits[row]
    if x.dtype == torch.bfloat16 and temp != 0.0:
        x = x.float() / torch.tensor(temp, dtype=torch.float32, device=x.device)
    return x.double()


def rejection_sample_reference(target_logits, draft_logits, draft_sampled, pos, idx_mapping, temperature, seed,
                               synthetic_rates=None):
    """(sampled int64 [B, 6], num_sampled int32 [B], margin fp32 [B]) of the module's contract."""
    b = idx_mapping.shape[0]
    assert target_logits.shape == (b * ROWS, V) and draft_logits.shape[1:] == (NSPEC, V)
    dev = target_logits.device
    sampled = torch.full((b, ROWS), -1, dtype=torch.int64)
    num_sampled = torch.zeros(b, dtype=torch.int32)
    margin = torch.full((b,), float("inf"), dtype=torch.float32)
    ds = draft_sampled.cpu().tolist()
    ps = pos.cpu().tolist()
    cols = torch.arange(V, device=dev)
    for r in range(b):
        slot = int(idx_mapping[r])
        temp = float(temperature[slot])
        sd = int(seed[slot]) & _M64
        base = r * ROWS
        acc = 0
        mg = float("inf")
        lp_t = lp_d = None
        for j in range(NSPEC):
            tok = ds[base + j + 1]
            u = philox_uniform(sd, ps[base + j])
            t = _target_row(target_logits, base + j, temp)
            ok = tok >= 0
            if temp == 0.0:
                if synthetic_rates is not None:
                    accept = ok and u < float(synthetic_rates[j])
                    mg = min(mg, abs(u - float(synthetic_rates[j])))
                else:
                    accept = ok and tok == _argmax_lowest(t)
            else:
                lp_t = t - torch.logsumexp(t, 0)
                d = draft_logits[slot, j].double() / float(np.float32(temp))
                lp_d = d - torch.logsumexp(d, 0)
                if synthetic_rates is not None:
                    accept = ok and u < float(synthetic_rates[j])
                    mg = min(mg, abs(u - float(synthetic_rates[j])))
                else:
                    if ok:
                        lhs, rhs = float(lp_t[tok]), float(np.log(np.float32(u))) + float(lp_d[tok])
                        accept = lhs > rhs
                        mg = min(mg, abs(lhs - rhs))
                    else:
                        accept = False
            if not accept:
                break
            acc += 1
        t = _target_row(target_logits, base + acc, temp)
        bonus = acc == NSPEC
        tok = -1 if bonus else ds[base + acc + 1]
        if temp == 0.0:
            token = _argmax_lowest(t)
        else:
            if bonus or tok < 0:
                res = t
            else:
                lp_t = t - torch.logsumexp(t, 0)
                d = draft_logits[slot, acc].double() / float(np.float32(temp))
                lp_d = d - torch.logsumexp(d, 0)
                ratio = torch.exp(lp_d - lp_t)
                res = torch.where(ratio < 1.0, lp_t + _log1p_neg_stable(ratio), torch.full_like(t, float("-inf")))
            val = res + gumbel_noise(sd, ps[base + acc], cols)
            token = _argmax_lowest(val)
            mg = min(mg, _top_gap(val)) if torch.isfinite(val.max()) else mg
        sampled[r, :acc] = torch.tensor(ds[base + 1:base + 1 + acc], dtype=torch.int64)
        sampled[r, acc] = token
        num_sampled[r] = acc + 1
        margin[r] = mg
    return sampled, num_sampled, margin
