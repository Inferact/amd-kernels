# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Routed-expert host: weight layouts, the token-count dispatch and the launch chain: sort (or block table) with the
optional bf16 quant, then GEMM1 + GEMM2 + combine in one launch."""
from dataclasses import dataclass

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from .kernel import (E, H, I, NE_MAX, READY_STRIDE, TOPK, compile_blocks, compile_combine, compile_experts,
                     compile_gemm1, compile_gemm2, compile_sort, max_blocks)

MAX_TOKENS = 1024
# upper token bound -> (bm, GEMM1 K-waves, GEMM1 prefetch, GEMM1 16-column pairs per wave, GEMM1 ring stages,
# GEMM2 columns, GEMM2 prefetch, GEMM2 ring stages, cached weight loads for experts with several blocks (1 GEMM2,
# 2 GEMM1, 3 both), waves per workgroup, tokens per combine workgroup): the fastest per stage of the sweep at 16-1024
# tokens (16 experts, 3 per token), uniform and skewed
_T0 = (16, 4, 3, 2, 0, 128, 4, 0, 1, 4, 1)
_T1 = (32, 4, 2, 2, 0, 256, 4, 3, 1, 4, 1)
_T2 = (64, 1, 2, 2, 4, 256, 3, 6, 1, 8, 2)
_T3 = (128, 1, 2, 2, 3, 256, 3, 3, 1, 8, 4)
DISPATCH = ((48, _T0), (100, _T1), (400, _T2), (MAX_TOKENS, _T3))
SORT_TIERS = (16, 32, 64, 128, 192, 256, 384, 512, 768, MAX_TOKENS)
WARM_UP_TOKENS = tuple(sorted(set(SORT_TIERS) | {bound for bound, _ in DISPATCH} | {1}))


def config_for(n_tokens: int):
    return next(cfg for bound, cfg in DISPATCH if n_tokens <= bound)


@dataclass
class ExpertWeights:
    """One layer's local experts (16 by default, up to ``NE_MAX``) in the kernel layouts (a pure
    permutation of the checkpoint bytes)."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor

    @property
    def n_experts(self) -> int:
        return self.w13.numel() // (2 * I * (H // 2))

    @classmethod
    def from_row_major(cls, w13, w13_scale, w2, w2_scale) -> "ExpertWeights":
        """From the checkpoint layouts of ``reference`` (uint8 views of the packed FP4 and e8m0 tensors); once, before
        graph capture. Same bytes as AITER's ``shuffle_weight_a16w4`` / ``shuffle_scale_a16w4``."""
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        n = w13.shape[0]
        assert 1 <= n <= NE_MAX and w13.shape[1:] == (2 * I, H // 2) and w2.shape[1:] == (H, I // 2)
        u8 = torch.uint8
        return cls(
            shuffle_weight_a16w4(w13.view(u8), 16, True).view(u8).contiguous(),
            shuffle_scale_a16w4(w13_scale.view(u8).reshape(-1, H // 32), n, True).view(u8).contiguous(),
            shuffle_weight_a16w4(w2.view(u8), 16, False).view(u8).contiguous(),
            shuffle_scale_a16w4(w2_scale.view(u8).reshape(-1, I // 32), n, False).view(u8).contiguous(),
        )


def _activation(x, x_scales):
    """(codes, scales, quantize): MXFP8 rows pass through, bf16 rows get quant workgroups in the routing launch."""
    if x.dtype == torch.bfloat16:
        assert x_scales is None and x.is_contiguous()
        dev = x.device
        return (torch.empty(x.shape, dtype=torch.uint8, device=dev),
                torch.empty((x.shape[0], H // 32), dtype=torch.uint8, device=dev), True)
    assert x_scales is not None and x.is_contiguous() and x_scales.is_contiguous()
    return x.view(torch.uint8), x_scales.view(torch.uint8), False


def moe_sort(topk_ids, topk_weights, block_m, x=None, xq=None, xs=None, ne=E):
    """Rows sorted by expert in blocks of ``block_m``: (ids, weights, block experts, [padded row count]). With bf16
    ``x`` the launch also quantizes its rows into ``xq`` / ``xs``."""
    n = topk_ids.shape[0]
    tier = next(t for t in SORT_TIERS if n <= t)
    ms = max_blocks(tier, block_m, ne) * block_m
    dev = topk_ids.device
    ids = torch.empty(ms, dtype=torch.int32, device=dev)
    wts = torch.empty(ms, dtype=torch.float32, device=dev)
    eids = torch.empty(ms // block_m, dtype=torch.int32, device=dev)
    nv = torch.empty(1, dtype=torch.int32, device=dev)
    quant = x is not None
    flat = topk_ids.contiguous().int().view(-1)
    _run_compiled(compile_sort(block_m=block_m, max_tokens=tier, quant=quant, ne=ne), flat,
                  topk_weights.contiguous().float().view(-1), ids, wts, eids, nv, x.view(-1) if quant else flat,
                  xq.view(-1) if quant else flat, xs.view(-1) if quant else flat, n, torch.cuda.current_stream())
    return ids, wts, eids, nv


def moe_blocks(expert_offsets, row_weights, block_m, x=None, xq=None, xs=None):
    """The block table of ``n_rows = row_weights.numel()`` input rows sorted by expert (``expert_offsets``, 49 int32:
    the first row of each expert, then the row count): as ``moe_sort`` with ``ids`` = the input rows."""
    n = row_weights.numel()
    ne = expert_offsets.numel() - 1
    tier = TOPK * next(t for t in SORT_TIERS if -(-n // TOPK) <= t)
    ms = (tier // block_m + ne) * block_m
    dev = row_weights.device
    ids = torch.empty(ms, dtype=torch.int32, device=dev)
    wts = torch.empty(ms, dtype=torch.float32, device=dev)
    eids = torch.empty(ms // block_m, dtype=torch.int32, device=dev)
    nv = torch.empty(1, dtype=torch.int32, device=dev)
    quant = x is not None
    off = expert_offsets.contiguous().int()
    exe = compile_blocks(block_m=block_m, max_rows=tier, quant=quant, ne=ne)
    _run_compiled(exe, off, row_weights.contiguous().float(), ids, wts, eids, nv, x.view(-1) if quant else off,
                  xq.view(-1) if quant else off, xs.view(-1) if quant else off, n, torch.cuda.current_stream())
    return ids, wts, eids, nv


def gemm1_swiglu_quant(x_codes, x_scales, w, ids, eids, nv, *, bm, kw, pf, rows, ni=2, nbuf=0, nwv=4, hot=False):
    """Sorted rows' clamp-SwiGLU MXFP8 intermediate in GEMM2's operand layout (``reference.unpack_rows`` gives
    ``[rows, 2304]`` e4m3 codes and ``[rows, 72]`` e8m0)."""
    n = x_codes.shape[0]
    q = torch.empty(rows * I, dtype=torch.uint8, device=x_codes.device)
    s = torch.empty(rows * (I // 32), dtype=torch.uint8, device=x_codes.device)
    ne = w.n_experts
    exe = compile_gemm1(kw=kw, pf=pf, bm=bm, ni=ni, nbuf=nbuf, nwv=nwv, ne=ne, hot=hot)
    _run_compiled(exe, x_codes.view(torch.uint8).view(-1),
                  x_scales.view(torch.uint8).view(-1), w.w13.view(-1), w.w13_scale.view(-1), q, s,
                  ids, eids, nv, n, rows, max_blocks(n, bm, ne), torch.cuda.current_stream())
    return q, s


def gemm2_slots(q, s, w, ids, wts, eids, nv, n_tokens, *, bm, tn, pf, nbuf=0, hot=False, nwv=4):
    """Route-weighted bf16 slots ``[n_tokens, 3, 5120]``."""
    out = torch.zeros((n_tokens, TOPK, H), dtype=torch.bfloat16, device=q.device)
    ne = w.n_experts
    exe = compile_gemm2(tn=tn, bm=bm, pf=pf, nbuf=nbuf, hot=hot, nwv=nwv, ne=ne)
    _run_compiled(exe, q, s, w.w2.view(-1), w.w2_scale.view(-1),
                  out.view(-1), ids, wts, eids, nv, n_tokens, q.numel() // I, max_blocks(n_tokens, bm, ne),
                  torch.cuda.current_stream())
    return out


def combine_slots(slots, topk_ids, ne=E):
    out = torch.empty((slots.shape[0], H), dtype=torch.bfloat16, device=slots.device)
    _run_compiled(compile_combine(ne=ne), topk_ids.contiguous().int().view(-1), slots.view(-1), out.view(-1),
                  slots.shape[0], torch.cuda.current_stream())
    return out


_arenas: dict = {}


def ready_counters(device):
    """The device's counters of the fused launch (``kernel.compile_experts``): zero between calls, one per process;
    two calls must not run concurrently on one device."""
    arena = _arenas.get(device)
    if arena is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the first call on a device allocates the ready counters: run it before graph capture")
        arena = _arenas[device] = torch.zeros((max_blocks(MAX_TOKENS, 16, NE_MAX) + 1) * READY_STRIDE // 4,
                                              dtype=torch.int32, device=device)
    return arena


def _fused(xq, xs, w, topk_ids, ids, wts, eids, nv, n_rows, nmb, *, bm, cfg, rows_out):
    """The fused launch on a routing table: the combined ``[n, 5120]`` bf16 rows, or with ``rows_out`` the
    route-weighted row of each input row."""
    _, kw, pf1, ni, nb1, tn, pf2, nb2, hot, nwv, cw = cfg
    rows = ids.numel()
    dev = xq.device
    q = torch.empty(rows * I, dtype=torch.uint8, device=dev)
    s = torch.empty(rows * (I // 32), dtype=torch.uint8, device=dev)
    slots = torch.empty(0 if rows_out else n_rows * TOPK * H, dtype=torch.bfloat16, device=dev)
    out = torch.empty((n_rows, H), dtype=torch.bfloat16, device=dev)
    exe = compile_experts(bm=bm, kw=kw, pf1=pf1, ni=ni, nb1=nb1, tn2=tn, pf2=pf2, nb2=nb2, hot=bool(hot & 1), nwv=nwv,
                          cw=cw, rows_out=rows_out, ne=w.n_experts, hot1=bool(hot & 2))
    _run_compiled(exe, xq.view(-1), xs.view(-1), w.w13.view(-1), w.w13_scale.view(-1), q, s, w.w2.view(-1),
                  w.w2_scale.view(-1), slots, out.view(-1), topk_ids, ids, wts, eids, nv, ready_counters(dev), n_rows,
                  rows, nmb, torch.cuda.current_stream())
    return out


def mxfp4_experts(x, x_scales, w: ExpertWeights, topk_weights, topk_ids, *, return_slots=False):
    """The 3 routed experts of each token. ``x``: MXFP8 e4m3 ``[n, 5120]`` with its e8m0 ``x_scales`` ``[n, 160]``, or
    bf16 ``[n, 5120]`` with ``x_scales`` None (quantized in the sort launch); ``topk_ids`` ``[n, 6]`` local expert ids,
    an id outside [0, ne) (padding row, expert of another rank) is not routed, ne = ``w.n_experts``;
    ``topk_weights`` FP32. Returns bf16 ``[n, 5120]``: the route-weighted slots summed in slot order, or with
    ``return_slots`` the slots ``[n, 3, 5120]`` (zero where not routed)."""
    n = x.shape[0]
    assert x.shape[1] == H and topk_ids.shape == (n, TOPK) and 1 <= n <= MAX_TOKENS
    cfg = config_for(n)
    bm = cfg[0]
    xq, xs, quant = _activation(x, x_scales)
    ne = w.n_experts
    ids, wts, eids, nv = moe_sort(topk_ids, topk_weights, bm, *((x, xq, xs) if quant else ()), ne=ne)
    if not return_slots:
        return _fused(xq, xs, w, topk_ids.contiguous().int().view(-1), ids, wts, eids, nv, n, max_blocks(n, bm, ne),
                      bm=bm, cfg=cfg, rows_out=False)
    _, kw, pf1, ni, nb1, tn, pf2, nb2, hot, nwv, _ = cfg
    q, s = gemm1_swiglu_quant(xq, xs, w, ids, eids, nv, bm=bm, kw=kw, pf=pf1, rows=ids.numel(), ni=ni, nbuf=nb1,
                              nwv=nwv, hot=bool(hot & 2))
    return gemm2_slots(q, s, w, ids, wts, eids, nv, n, bm=bm, tn=tn, pf=pf2, nbuf=nb2, hot=bool(hot & 1), nwv=nwv)


def mxfp4_experts_rows(x, x_scales, w: ExpertWeights, row_weights, expert_offsets):
    """The expert of each row of rows already sorted by expert (the receive buffer of an expert-parallel dispatch):
    ``x`` MXFP8 e4m3 ``[R, 5120]`` + ``x_scales`` e8m0 ``[R, 160]``, or bf16 with ``x_scales`` None; expert ``e`` owns
    rows ``expert_offsets[e]`` .. ``expert_offsets[e + 1]`` (ne + 1 int32, the last the row count, at most ``R``);
    ``row_weights`` FP32 ``[R]``. Returns bf16 ``[R, 5120]``: row ``r`` = ``row_weights[r] * expert(x[r])`` rounded
    once; rows from ``expert_offsets[ne]`` on are not written."""
    n = x.shape[0]
    assert x.shape[1] == H and row_weights.shape == (n,) and expert_offsets.shape == (w.n_experts + 1,)
    assert 1 <= n <= TOPK * MAX_TOKENS
    cfg = config_for(-(-n // TOPK))
    bm = cfg[0]
    xq, xs, quant = _activation(x, x_scales)
    ids, wts, eids, nv = moe_blocks(expert_offsets, row_weights, bm, *((x, xq, xs) if quant else ()))
    return _fused(xq, xs, w, ids, ids, wts, eids, nv, n, n // bm + min(w.n_experts, n), bm=bm, cfg=cfg,
                  rows_out=True)


def _mode(x, x_scales, *args, **kwargs):
    """What a call fixes about its kernels besides the row count (``amd_kernels.warm_up``): bf16 or MXFP8 rows,
    slots or the combined rows, the local expert count."""
    return x.dtype == torch.bfloat16, bool(kwargs.get("return_slots", False)), args[0].n_experts


def _inputs(mode, n, x, x_scales, w, topk_weights, topk_ids, **kwargs):
    dev = x.device
    xn = torch.zeros((n, H), dtype=x.dtype, device=dev)
    sn = None if x_scales is None else torch.zeros((n, H // 32), dtype=x_scales.dtype, device=dev)
    return (xn, sn, w, torch.ones((n, TOPK), device=dev), torch.zeros((n, TOPK), dtype=torch.int32, device=dev)), kwargs


def _rows_inputs(mode, n, x, x_scales, w, row_weights, expert_offsets):
    dev = x.device
    xn = torch.zeros((n, H), dtype=x.dtype, device=dev)
    sn = None if x_scales is None else torch.zeros((n, H // 32), dtype=x_scales.dtype, device=dev)
    return (xn, sn, w, torch.ones(n, device=dev), torch.zeros(w.n_experts + 1, dtype=torch.int32, device=dev)), {}


mxfp4_experts = entry("moe/mxfp4_h5120_i2304_e16_top3", mode=_mode, sizes=lambda mode: WARM_UP_TOKENS,
                      inputs=_inputs)(mxfp4_experts)
mxfp4_experts_rows = entry("moe/mxfp4_h5120_i2304_e16_top3_rows",
                           mode=lambda x, x_scales, *args: (x.dtype == torch.bfloat16, args[0].n_experts),
                           sizes=lambda mode: tuple(TOPK * t for t in WARM_UP_TOKENS), inputs=_rows_inputs)(
    mxfp4_experts_rows)
