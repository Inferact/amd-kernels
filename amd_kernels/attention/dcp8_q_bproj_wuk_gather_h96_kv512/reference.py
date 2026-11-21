# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gathered latent query of the DCP8 DSA decode path.

Inputs: ``q_a`` bf16 [M, 1536]; per rank r in 0..7 the q_b weight ``w_qb[r]`` bf16 [2304, 1536] (12 heads x (128 nope
+ 64 rope-width pass-through) rows, head-major) and ``w_uk[r]`` bf16 [12, 128, 512] (W_UK_T of its 12 heads).
Output: bf16 [M, 96, 576], head 12 r + h from rank r: ``[bf16(q_nope @ w_uk) | q_pe]`` with
``q = bf16(q_a @ w_qb.T)`` and ``q_nope``, ``q_pe`` its first 128 and last 64 columns per head. Both matmuls
accumulate in fp32 and round once to bf16, as the unfused vLLM path (hipBLASLt GEMM, then ``torch.bmm``, then
concat, then the all-gather) does. A kernel sums K in a different order, so a q element may differ by one bf16 ulp
and a latent element then follows: the tests require per-row cosine >= 0.9999 and relative L2 <= 4e-3 against this
function, bitwise repeatability of two runs, and exact equality of the q_pe columns.
"""

import torch

HEADS_PER_RANK = 12
RANKS = 8


def q_b_reference(q_a, w_qb):
    return (q_a.float() @ w_qb.float().T).bfloat16().view(q_a.shape[0], HEADS_PER_RANK, 192)


def latent_reference(q, w_uk):
    nope = q[..., :128].transpose(0, 1).float()
    lat = torch.bmm(nope, w_uk.float()).bfloat16().transpose(0, 1)
    return torch.cat((lat, q[..., 128:]), dim=-1)


def gathered_query_reference(q_a, w_qb, w_uk):
    """``w_qb``, ``w_uk``: the 8 ranks' weights (sequences); returns bf16 [M, 96, 576]."""
    return torch.cat([latent_reference(q_b_reference(q_a, w_qb[r]), w_uk[r]) for r in range(RANKS)], dim=1)


L_LATENT, ROPE, BLOCK = 512, 64, 128
Q_RECORD_BYTES, Q_HI, Q_LO, Q_ROPE, Q_SCALE = 1168, 0, 512, 1024, 1152


def quantize_q(q):
    """The fp8 hi/lo record of the sparse-MLA decode kernel (``quantize_q`` of
    attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048/reference.py) from bf16 ``q [rows, 96, 576]``: uint8
    ``[rows, 96, 1168]``. Per (row, head) and block b of 128 latent dims ``s_b = max(exponent(amax_b) - 7, 1)`` (the E8M0
    byte, ``exponent`` the biased f32 exponent), ``x = q * 2^(127 - s_b)``, hi = e4m3fn(x) (round to nearest even),
    lo = e4m3fn((x - hi) * 32). Bytes: hi [0, 512), lo [512, 1024), rope bf16 [1024, 1152), s_0..s_3 [1152, 1156), zero
    padding to 1168."""
    rows, heads, _ = q.shape
    x = q[..., :L_LATENT].float().view(rows, heads, 4, BLOCK)
    amax = x.abs().amax(dim=-1)
    e = (amax.view(torch.int32) >> 23) & 0xFF
    s = (e - 7).clamp(min=1)
    inv = ((254 - s) << 23).to(torch.int32).view(torch.float32)
    xs = x * inv[..., None]
    hi = xs.to(torch.float8_e4m3fn)
    lo = ((xs - hi.float()) * 32.0).to(torch.float8_e4m3fn)
    out = torch.zeros(rows, heads, Q_RECORD_BYTES, dtype=torch.uint8, device=q.device)
    out[..., Q_HI:Q_LO] = hi.view(torch.uint8).view(rows, heads, L_LATENT)
    out[..., Q_LO:Q_ROPE] = lo.view(torch.uint8).view(rows, heads, L_LATENT)
    out[..., Q_ROPE:Q_SCALE] = q[..., L_LATENT:].contiguous().view(torch.uint8).view(rows, heads, 2 * ROPE)
    out[..., Q_SCALE : Q_SCALE + 4] = s.to(torch.uint8)
    return out


def dequantize_q(q8):
    """The bf16-valued f32 ``[rows, 96, 576]`` a consumer reconstructs from ``quantize_q``'s record: latent
    ``(hi + lo / 32) * 2^(s_b - 127)``, rope unchanged."""
    rows, heads, _ = q8.shape
    hi = q8[..., Q_HI:Q_LO].contiguous().view(torch.float8_e4m3fn).float().view(rows, heads, 4, BLOCK)
    lo = q8[..., Q_LO:Q_ROPE].contiguous().view(torch.float8_e4m3fn).float().view(rows, heads, 4, BLOCK)
    s = q8[..., Q_SCALE : Q_SCALE + 4].int()
    scale = ((s << 23).to(torch.int32).view(torch.float32))[..., None]
    lat = ((hi + lo / 32.0) * scale).view(rows, heads, L_LATENT)
    rope = q8[..., Q_ROPE:Q_SCALE].contiguous().view(torch.bfloat16).float()
    return torch.cat((lat, rope), dim=-1)
