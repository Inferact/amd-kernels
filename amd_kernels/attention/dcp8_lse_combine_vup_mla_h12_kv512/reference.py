# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP8 LSE combine fused with the W_UV up-projection (v_up first), the contract of the kernel.

Every rank r holds the partial latent attention output ``x_r [T, 96, 512]`` (bf16) and the partial LSE ``lse_r [T, 96]``
of its KV shard for all 96 heads. Rank d keeps heads 12 d .. 12 d + 11 of the combined, up-projected output.

Per rank and head: ``v_r = bf16(x_r[t, h] @ W_UV[h])`` (fp32 accumulate, ``W_UV [96, 512, 128]``). The LSE is masked to
-inf for rows past ``query_start_loc[num_seqs]`` and rows of a request whose ``seq_lens`` is 0 (no local shard), and
NaN / +inf to -inf. The weight of source s is ``exp(lse_s - m) / sum_r exp(lse_r - m)`` with ``m = max_r lse_r`` (0 when
all are -inf); a NaN weight (a row empty on every rank) is 0 and a zero weight drops the source's value (NaN
partials of empty shards never propagate). ``out[t, j] = bf16(sum_s w_s * v_s)``: fp32 products summed in rank order,
rounded once. The optional output gate then multiplies by ``sigmoid(gate)`` in fp32 and rounds again.

Tolerance: the kernel's ``v_r`` accumulates K in four fp32 partial sums (one per wave) added in fixed order, so it may
differ from the reference's single accumulation by one bf16 ulp of ``v``, and it multiplies by the hardware reciprocal
of the sum instead of dividing; ``out`` agrees to cosine > 0.9999 and
max-abs error <= 2^-7 of the row max, and two runs match bit for bit.
"""

import math

import torch

H = 96
L = 512
V = 128
WORLD = 8
HPR = H // WORLD


def row_mask(num_rows, seq_lens, query_start_loc, device):
    """bool [num_rows]: rows whose local shard is non-empty."""
    valid = torch.ones(num_rows, dtype=torch.bool, device=device)
    if seq_lens is None:
        return valid
    qsl = query_start_loc.tolist()
    n = seq_lens.shape[0]
    valid[qsl[n]:] = False
    for i, sl in enumerate(seq_lens.tolist()):
        if sl == 0:
            valid[qsl[i]:qsl[i + 1]] = False
    return valid


def partial_v(x, w_uv):
    """bf16 [T, n, 128]: ``x [T, n, 512] @ w_uv [n, 512, 128]`` per head, fp32 accumulate."""
    return torch.bmm(x.float().transpose(0, 1), w_uv.float()).transpose(0, 1).to(torch.bfloat16)


def combine_reference(xs, lses, w_uv, rank, seq_lens=None, query_start_loc=None, is_lse_base_on_e=True, gate=None):
    """out bf16 [T, 12, 128] of ``rank``'s heads from the 8 ranks' partials ``xs`` (each [T, >=96, 512] bf16) and
    ``lses`` (each [T, >=96] fp32); ``seq_lens`` / ``query_start_loc`` are one list per rank (or None). With ``gate``
    ([T, 1536] bf16, the DSA output gate) the result is ``bf16(out * sigmoid(gate))`` in fp32."""
    t = xs[0].shape[0]
    sl = slice(rank * HPR, (rank + 1) * HPR)
    vs, ls = [], []
    for r in range(WORLD):
        vs.append(partial_v(xs[r][:, sl], w_uv[sl]).float())
        lse = lses[r][:, sl].float()
        if seq_lens is not None:
            lse = lse.masked_fill(~row_mask(t, seq_lens[r], query_start_loc[r], lse.device)[:, None], -math.inf)
        ls.append(torch.where(lse.isnan() | (lse == math.inf), -math.inf, lse))
    lse = torch.stack(ls)  # [8, T, 12]
    m = lse.amax(0)
    m = torch.where(m == -math.inf, torch.zeros_like(m), m)
    e = torch.exp(lse - m) if is_lse_base_on_e else torch.exp2(lse - m)
    w = e / e.sum(0)
    w = torch.where(w.isnan(), torch.zeros_like(w), w)
    acc = torch.zeros_like(vs[0])
    for r in range(WORLD):
        acc = acc + torch.where(w[r][..., None] == 0, torch.zeros_like(vs[r]), vs[r] * w[r][..., None])
    out = acc.to(torch.bfloat16)
    if gate is not None:
        out = (out.float() * torch.sigmoid(gate.float()).view(t, HPR, V)).to(torch.bfloat16)
    return out
