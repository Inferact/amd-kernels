# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Torch reference of the expert-parallel routed experts: the contract the tests check (never called in serving).

The single-GPU routed-experts reference (below) is the contract of one rank's experts; ``ep_reference`` applies it to
the experts of every rank and sums a token's six slots in slot order. Expert ids are global: expert ``g`` lives on rank
``g // 48`` as its local expert ``g % 48``.

Operands. Activations MXFP8: e4m3 codes ``[M, 5120]`` with one e8m0 byte per 32 columns (row-major ``[M, 160]``).
Weights MXFP4: e2m1 codes packed two per byte along K (low nibble first) with one e8m0 per 32 K; checkpoint layouts
``w13 [E, 2*2304, 2560]`` (gate rows, then up rows), ``w13_scale [E, 4608, 160]``, ``w2 [E, 5120, 1152]``,
``w2_scale [E, 5120, 72]``, E = 48 local experts. The kernels read them in the layouts of AITER's
``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4`` (``w13_layout`` and ``w2_layout`` here are the weight permutations).

Numerics, per (token, slot) route of expert e: ``gate, up = x @ w13[e].T`` in fp32; ``g = min(gate, limit)``,
``u = clamp(up, -limit, limit)``, ``act = silu(g) * u`` in fp32 (limit 10); ``act`` is quantized to MXFP8 per 32
columns: e8m0 = the exponent of ``ceil_pow2(amax * fp32(1 / 448))`` (at least 1) and e4m3 codes (nearest even,
saturating) of ``act * 2^(127 - e8m0)``; ``slot = bf16(w_route * (q @ w2[e].T))``. ``out[t] = bf16(sum over the 6
slots in slot order, fp32)``. The kernels accumulate in another order, so a bf16 slot or a code of ``act`` may differ
near a rounding tie (4 % of the slots differ in one bf16 ulp): the tests bound the cosine over the output >= 0.99999
(each row >= 0.9999) and check two runs, and a graph replay, bit for bit.
"""

import torch

H, I, E, TOPK = 5120, 2304, 48, 6
LIMIT = 10.0
_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def dequant_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Row-major MXFP4 (``[.., K/2]`` u8, ``[.., K/32]`` e8m0 u8) -> f32 ``[.., K]``."""
    lut = torch.cat([_GRID, -_GRID]).to(codes.device)
    v = torch.stack([lut[(codes & 0xF).long()], lut[(codes >> 4).long()]], dim=-1).flatten(-2)
    return v * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=-1)


def dequant_mxfp8(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """e4m3 ``[M, K]`` + e8m0 ``[M, K/32]`` -> f32 ``[M, K]``."""
    m, k = codes.shape
    v = codes.float().reshape(m, k // 32, 32)
    return (v * torch.exp2(scales.float() - 127.0).unsqueeze(-1)).reshape(m, k)


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """f32 / bf16 ``[M, K]`` -> e4m3 codes ``[M, K]``, e8m0 bytes ``[M, K/32]`` (the exponent of
    ``ceil_pow2(amax * fp32(1 / 448))``, at least 1)."""
    m, k = x.shape
    blocks = x.float().reshape(m, k // 32, 32)
    u = (blocks.abs().amax(-1) * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    e = torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp_min(1)
    codes = (blocks * torch.exp2(127.0 - e.float()).unsqueeze(-1)).clamp(-448.0, 448.0).reshape(m, k)
    return codes.to(torch.float8_e4m3fn), e.to(torch.uint8)


def clamp_silu_mul(gate: torch.Tensor, up: torch.Tensor, limit: float = LIMIT) -> torch.Tensor:
    g = gate.float().clamp(max=limit)
    return torch.nn.functional.silu(g) * up.float().clamp(-limit, limit)


def experts_reference(
    x_codes, x_scales, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids, *, return_slots=False
) -> torch.Tensor:
    """bf16 ``[M, 5120]``: the weighted sum of each token's 6 experts (checkpoint-layout weights, module docstring; an
    id outside [0, 48) is not routed and contributes zero); ``return_slots``: the bf16 ``[M, 6, 5120]`` slots
    instead."""
    m = x_codes.shape[0]
    x = dequant_mxfp8(x_codes, x_scales)
    slots = torch.zeros((m, topk_ids.shape[1], w2.shape[1]), dtype=torch.bfloat16, device=x.device)
    for e in topk_ids.unique().tolist():
        if not 0 <= e < w13.shape[0]:
            continue
        t, s = torch.nonzero(topk_ids == e, as_tuple=True)
        gu = x[t] @ dequant_mxfp4(w13[e], w13_scale[e]).t()
        q = dequant_mxfp8(*quantize_mxfp8(clamp_silu_mul(gu[:, :I], gu[:, I:])))
        y = q @ dequant_mxfp4(w2[e], w2_scale[e]).t()
        slots[t, s] = (y * topk_weights[t, s].float().unsqueeze(1)).bfloat16()
    if return_slots:
        return slots
    return _ordered_sum(slots)


def experts_rows_reference(
    x_codes, x_scales, w13, w13_scale, w2, w2_scale, row_weights, expert_offsets
) -> torch.Tensor:
    """bf16 ``[R, 5120]``: row ``r`` of expert ``e`` (rows ``expert_offsets[e]`` .. ``expert_offsets[e + 1]``) is
    ``bf16(row_weights[r] * (the expert of x[r]))``, numerics as ``experts_reference``'s slot; rows past
    ``expert_offsets[48]`` are zero."""
    x = dequant_mxfp8(x_codes, x_scales)
    out = torch.zeros((x.shape[0], w2.shape[1]), dtype=torch.bfloat16, device=x.device)
    offs = expert_offsets.tolist()
    for e in range(len(offs) - 1):
        if offs[e + 1] == offs[e]:
            continue
        r = slice(offs[e], offs[e + 1])
        gu = x[r] @ dequant_mxfp4(w13[e], w13_scale[e]).t()
        q = dequant_mxfp8(*quantize_mxfp8(clamp_silu_mul(gu[:, :I], gu[:, I:])))
        out[r] = ((q @ dequant_mxfp4(w2[e], w2_scale[e]).t()) * row_weights[r].float().unsqueeze(1)).bfloat16()
    return out


def _ordered_sum(slots: torch.Tensor) -> torch.Tensor:
    acc = slots[:, 0].float()
    for s in range(1, slots.shape[1]):
        acc = acc + slots[:, s].float()
    return acc.bfloat16()


def w13_layout(w13: torch.Tensor) -> torch.Tensor:
    """``[E, 2I, H/2]`` u8 (gate rows, then up rows) -> ``[E, I/16, 2 (gate, up), H/128, klane 4, nlane 16, 16 B]``."""
    e, n2, half_k = w13.shape
    v = w13.view(e, 2, n2 // 32, 16, half_k // 64, 4, 16)
    return v.permute(0, 2, 1, 4, 5, 3, 6).contiguous()


def w2_layout(w2: torch.Tensor) -> torch.Tensor:
    """``[E, H, I/2]`` u8 -> ``[E, H/16, I/128, klane 4, nlane 16, 16 B]``."""
    e, n, half_k = w2.shape
    return w2.view(e, n // 16, 16, half_k // 64, 4, 16).permute(0, 1, 3, 4, 2, 5).contiguous()


def unpack_rows(q: torch.Tensor, s: torch.Tensor, rows: int, k: int = I) -> tuple[torch.Tensor, torch.Tensor]:
    """The MXFP8 operand layout of the sorted rows (GEMM2's input, GEMM1's gathered input) -> row-major codes
    ``[rows, k]`` and e8m0 ``[rows, k / 32]``. Per 16-row block and 128-K step: ``[half 2][lane 64][16 B]`` with lane =
    16 * (K % 64 // 16) + row and half = K // 64 % 2; the e8m0 words (4 bytes = 4 K groups) as ``[step][row]``."""
    steps = k // 128
    c = q.view(rows // 16, steps, 2, 4, 16, 16).permute(0, 4, 1, 2, 3, 5).reshape(rows, k)
    return c, s.view(rows, k // 32)


def ep_reference(x_codes, x_scales, weights, topk_weights, topk_ids) -> torch.Tensor:
    """bf16 ``[M, 5120]`` of a source rank's ``M`` tokens: ``weights`` is one ``(w13, w13_scale, w2, w2_scale)`` tuple of
    checkpoint-layout tensors per rank (equal expert counts), ``topk_ids`` are global expert ids (an id outside [0, experts * len(weights)) is
    not routed and contributes zero)."""
    slots = None
    for owner, w in enumerate(weights):
        local = topk_ids - owner * w[0].shape[0]
        part = experts_reference(x_codes, x_scales, *w, topk_weights, local, return_slots=True)
        slots = part if slots is None else slots + part
    return _ordered_sum(slots)
