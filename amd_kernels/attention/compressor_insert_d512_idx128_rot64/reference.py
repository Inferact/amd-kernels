# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch contract of the compressor tail and the indexer key store.

Inputs per token ``t``: the compressor GEMM's fp32 row ``[kv | score]`` (ratio 2) or ``kv`` (ratio 1), the position, the
ring slot (ratio 2; ratio 1 takes the main-cache slot), the main-cache slot and the indexer-cache slot (< 0 skips).
A token closes a group when ``(position + 1) % ratio == 0`` and its ring slot is >= 0.

Latent (fp32 until the end): ratio 1 pools nothing; ratio 2 pools the previous row p (the ring row ``(position - 1) %
ring_rows`` of the request's block when ``t`` is its request's first token, else the raw row ``t - 1``) and this row c
per dimension: ``peak = max(s_p, s_c)``, ``w = exp2((s - peak) * log2 e)``, ``(kv_p w_p + kv_c w_c) / (w_p + w_c)``. The
latent is ``bf16((pooled * rsqrt(mean(pooled^2) + eps)) * norm_weight)``; mean(pooled^2) sums the squares of 8
contiguous dims per lane in order, then the 64 lanes by ``wave_sum`` below.

Main cache: GPT-J RoPE of dims 448..511 of the latent (cos | sin of ``cos_sin_cache[position // ratio * ratio]``, fp32,
rounded to bf16), times ``1 / kv_scale``, clamped to +-448, e4m3 (RNE) into row ``slot % rows`` of page ``slot // rows``.

Indexer key: ``k_pre = bf16(latent @ wk^T)`` (an exact fp64 sum in the contract; the kernel's fp32 MFMA sum rounds
differently to bf16 in about 1 in 1000 elements), ``k = bf16((k_pre * rsqrt(mean(k_pre^2) + k_eps)) * k_norm_weight)``,
RoPE of dims 64..127 as above (the indexer's cos | sin cache), MXFP4 per 32 dims as in ``indexer/q_rope_mxfp4_quant``.
mean(k_pre^2) sums a lane's 4 columns (4g + i of a 16-column tile) in order, then lanes ``(g0 + g2) + (g1 + g3)``,
then the eight tiles ``((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7))``. Cache layout (page of ``E`` tokens, 64 value bytes
and 4 scale bytes each): value byte b of token p at ``p // 32 * 2048 + b // 16 * 512 + p % 32 * 16 + b % 16``, scale j at
``E * 64 + p // 32 * 128 + j % 2 * 64 + p % 32 * 2 + j // 2``.

Ring (ratio 2): each request's first token ``qs`` (``query_start_loc``) stores its request's last ``min(rows, ring_rows)``
raw rows to their ring slots (slot >= 0), after the ring reads above. Bit for bit except the ``k_pre`` rounding.
"""

import torch

LOG2E = torch.tensor(1.4426950408889634, dtype=torch.float32)
D = 512
KN = 128


def wave_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim (64 lanes) in the kernel's order: quads, half-rows, rows, then lanes ^ 32, ^ 16."""
    q = x.reshape(*x.shape[:-1], 16, 4)
    q = (q[..., 0] + q[..., 1]) + (q[..., 2] + q[..., 3])
    r = q.reshape(*x.shape[:-1], 4, 4)
    r = (r[..., 0] + r[..., 1]) + (r[..., 2] + r[..., 3])
    return (r[..., 0] + r[..., 2]) + (r[..., 1] + r[..., 3])


def lane_sq_sum(x: torch.Tensor) -> torch.Tensor:
    """Squares of consecutive 8-dim lanes of ``x [n, 512]`` summed in order, then ``wave_sum``."""
    sq = (x * x).reshape(x.shape[0], 64, 8)
    s = sq[..., 0]
    for i in range(1, 8):
        s = s + sq[..., i]
    return wave_sum(s)


def rope_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int) -> torch.Tensor:
    """bf16-rounded GPT-J rotation of the last ``rope_dim`` dims of fp32 ``x [n, d]``; cos / sin ``[n, rope_dim / 2]``."""
    out = x.clone()
    lo = x.shape[-1] - rope_dim
    even, odd = x[..., lo::2], x[..., lo + 1 :: 2]
    out[..., lo::2] = (even * cos - odd * sin).bfloat16().float()
    out[..., lo + 1 :: 2] = (odd * cos + even * sin).bfloat16().float()
    return out


def mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(packed u8 ``[n, d / 2]``, e8m0 u8 ``[n, d / 32]``), the scheme of ``indexer/q_rope_mxfp4_quant``."""
    g = x.reshape(x.shape[0], -1, 32)
    amax = g.abs().amax(-1, keepdim=True)
    floor = torch.tensor(6 * 2.0**-126, dtype=torch.float32, device=x.device)
    ratio = torch.maximum(amax, floor) * torch.tensor(1.0 / 6.0, dtype=torch.float32, device=x.device)
    e8m0 = (ratio.view(torch.int32) + 0x7FFFFF) >> 23
    qx = g * ((254 - e8m0) << 23).view(torch.float32)
    a = qx.abs()
    code = sum(
        (a >= t) if tie_up else (a > t)
        for t, tie_up in ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
    ).to(torch.int32)
    code = (code + 8 * (qx.view(torch.int32) < 0).to(torch.int32)).reshape(x.shape[0], -1, 2)
    return (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8).reshape(x.shape[0], -1), e8m0.reshape(x.shape[0], -1).to(
        torch.uint8
    )


def _cs(cache: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c = cache[rows].float()
    return c[:, :32], c[:, 32:]


def compressor_insert_ref(
    raw: torch.Tensor,  # [n, 512 * ratio] fp32
    positions: torch.Tensor,  # [n] int64
    ring: torch.Tensor | None,  # [blocks, ring_rows, 1024] fp32 (ratio 2)
    ring_slots: torch.Tensor,  # [n] int64 (ratio 1: the main slots)
    query_start_loc: torch.Tensor | None,  # [reqs + 1] int32 (ratio 2)
    token_to_req: torch.Tensor | None,  # [n] int32 (ratio 2)
    norm_weight: torch.Tensor,  # [512] bf16
    eps: float,
    main_cache: torch.Tensor,  # [pages, rows, 512] e4m3
    main_slots: torch.Tensor,  # [n] int64
    cos_sin_main: torch.Tensor,  # [rows, 64] fp32
    kv_scale: torch.Tensor,  # fp32 scalar
    wk: torch.Tensor,  # [128, 512] bf16
    k_norm_weight: torch.Tensor,  # [128] bf16
    k_eps: float,
    idx_cache: torch.Tensor,  # u8 pages of E * 68 bytes
    idx_slots: torch.Tensor,  # [n] int64
    cos_sin_idx: torch.Tensor,  # [rows, 64] fp32 or bf16
    ratio: int,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Returns copies of (ring, main_cache, idx_cache) with this step's writes applied."""
    n = raw.shape[0]
    dev = raw.device
    t = torch.arange(n, device=dev)
    live = ((positions + 1) % ratio == 0) & (ring_slots >= 0)
    ring_out = None if ring is None else ring.clone()
    if ratio == 2:
        cap = ring.shape[1]
        req = token_to_req.long()
        qs, qe = query_start_loc.long()[req], query_start_loc.long()[req + 1]
        leader = t == qs
        slot = ring_slots.clamp(min=0)
        ring_prev = ring[slot // cap, (positions - 1) % cap]
        prev = torch.where(leader[:, None], ring_prev, raw[(t - 1).clamp(min=0)])
        kv_p, s_p, kv_c, s_c = prev[:, :D], prev[:, D:], raw[:, :D], raw[:, D:]
        peak = torch.maximum(s_p, s_c)
        w_p, w_c = torch.exp2((s_p - peak) * LOG2E.to(dev)), torch.exp2((s_c - peak) * LOG2E.to(dev))
        pooled = (kv_p * w_p + kv_c * w_c) / (w_p + w_c)
    else:
        pooled = raw[:, :D].clone()
    rstd = torch.rsqrt(lane_sq_sum(pooled) * (1.0 / D) + torch.tensor(eps, dtype=torch.float32, device=dev))
    latent = ((pooled * rstd[:, None]) * norm_weight.float()).bfloat16()
    lat = latent.float()

    pg = positions // ratio * ratio
    cos, sin = _cs(cos_sin_main, pg)
    rot = rope_pairs(lat, cos, sin, 64)
    rows = (rot * (1.0 / kv_scale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    main_out = main_cache.clone()
    ok = live & (main_slots >= 0)
    bs = main_cache.shape[1]
    flat = main_out.view(torch.uint8)
    flat[main_slots[ok] // bs, main_slots[ok] % bs] = rows[ok].view(torch.uint8)

    kpre = (latent.double() @ wk.double().T).float().bfloat16().float()
    sq = kpre.reshape(n, 8, 4, 4) ** 2  # (16-column tile, g, i)
    s = sq[..., 0]
    for i in range(1, 4):
        s = s + sq[..., i]
    tsum = (s[..., 0] + s[..., 2]) + (s[..., 1] + s[..., 3])
    total = ((tsum[:, 0] + tsum[:, 1]) + (tsum[:, 2] + tsum[:, 3])) + ((tsum[:, 4] + tsum[:, 5]) + (tsum[:, 6] + tsum[:, 7]))
    krstd = torch.rsqrt(total * (1.0 / KN) + torch.tensor(k_eps, dtype=torch.float32, device=dev))
    k = ((kpre * krstd[:, None]) * k_norm_weight.float()).bfloat16().float()
    cos, sin = _cs(cos_sin_idx, pg)
    krot = rope_pairs(k, cos, sin, 64)
    packed, scale = mxfp4_quant(krot)
    idx_out = idx_cache.clone()
    pages = idx_out.view(idx_out.shape[0], -1)
    e = pages.shape[1] // 68
    ok = live & (idx_slots >= 0)
    for tok in ok.nonzero().flatten().tolist():
        sl = int(idx_slots[tok])
        pi, p = sl // e, sl % e
        b = torch.arange(64, device=dev)
        pages[pi, p // 32 * 2048 + b // 16 * 512 + p % 32 * 16 + b % 16] = packed[tok]
        j = torch.arange(4, device=dev)
        pages[pi, e * 64 + p // 32 * 128 + j % 2 * 64 + p % 32 * 2 + j // 2] = scale[tok]

    if ratio == 2:
        for r in range(query_start_loc.numel() - 1):
            a, b = int(query_start_loc[r]), int(query_start_loc[r + 1])
            if a >= b or int(ring_slots[a]) < 0:
                continue
            for tok in range(max(a, b - cap), b):
                sl = int(ring_slots[tok])
                if sl >= 0:
                    ring_out[sl // cap, sl % cap] = raw[tok]
    return ring_out, main_out, idx_out
