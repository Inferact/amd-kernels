# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the routed experts: weights, workspace, launches, the warm-up."""
import dataclasses
import functools

import torch

from amd_kernels.warm_up import entry

from .common import E, H, I, TOPK
from .fused import compile_fused
from .gemm1 import compile_gemm1
from .gemm2 import compile_gemm2
from .quant import compile_quant
from .reduce import compile_reduce

MAX_M = 192
BM = 16
NB1, NB2 = 12, 14


def sort_capacity(m: int) -> int:
    """Rows of the BM=16 sort of ``m`` tokens (aiter's ``aux_sort_capacity``): every route plus 15 pad rows per active expert."""
    routes = m * TOPK
    return (routes + min(E, routes) * (BM - 1) + BM - 1) // BM * BM


@dataclasses.dataclass
class MoeWeights:
    """The routed experts of one layer in the layout vLLM / aiter hold (preshuffled): ``w13`` u8 [E, 768, 1792]
    (gate rows then up rows, ``aiter.shuffle_weight``), ``s13`` u8 [E * 768, 112] (``e8m0_shuffle``), ``w2`` u8
    [E, 3584, 192], ``s2`` u8 [E * 3584, 16]."""

    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor

    @classmethod
    def from_checkpoint(cls, w13, s13, w2, s2) -> "MoeWeights":
        """From the checkpoint's row-major e2m1 pairs ``[E, 768, 1792]`` / ``[E, 3584, 192]`` and e8m0 scales
        ``[E, 768, 112]`` / ``[E, 3584, 12]`` (u8): the aiter shuffles, once, before graph capture."""
        from aiter.ops.shuffle import shuffle_weight
        from aiter.utility.fp4_utils import e8m0_shuffle

        fp4 = torch.float4_e2m1fn_x2
        return cls.from_shuffled(shuffle_weight(w13.view(fp4)), e8m0_shuffle(s13.reshape(E * 2 * I, -1)),
                                 shuffle_weight(w2.view(fp4)), e8m0_shuffle(s2.reshape(E * H, -1)))

    @classmethod
    def from_shuffled(cls, w13, s13, w2, s2) -> "MoeWeights":
        """Views of tensors that already hold the aiter layouts (any 1-byte dtype)."""
        u8 = torch.uint8
        out = cls(w13.view(u8), s13.view(u8), w2.view(u8), s2.view(u8))
        assert out.w13.numel() == E * 2 * I * (H // 2) and out.w2.numel() == E * H * (I // 2)
        assert out.s13.numel() == E * 2 * I * (H // 32) and out.s2.numel() == E * H * 16
        for t in (out.w13, out.s13, out.w2, out.s2):
            assert t.is_contiguous()
        return out


@dataclasses.dataclass
class Sort:
    """The BM=16 sort of one call (``aiter.moe_sorting`` aux layout, or the router entry's ``make_aux``): rows of 16 in
    expert order, ``token | slot << 24`` per row (pad rows token = M), the router weight per row, the expert of every
    16-row block, ``num_valid_ids[0]`` = padded rows, and ``reverse_sorted[t * 16 + k]`` = the row of the pair."""

    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    reverse_sorted: torch.Tensor

    @classmethod
    def from_aux(cls, aux: dict) -> "Sort":
        return cls(aux["sorted_token_ids"], aux["sorted_weights"], aux["sorted_expert_ids"], aux["num_valid_ids"],
                   aux["reverse_sorted"])


def moe_sort(topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> Sort:
    """The sort by aiter's ``moe_sorting`` (for tests and benchmarks; serving takes the router entry's sort)."""
    import aiter.fused_moe as fm

    r = fm.moe_sorting(topk_ids.to(torch.int32), topk_weights.to(torch.float32), E, H, torch.bfloat16, BM, None, None,
                       output_aux=fm.AUX_SORT_OPUS, accumulate=True)
    return Sort(r[0], r[1], r[2], r[3], r[6])


_WORK: dict = {}


def _workspace(device: torch.device, stream: int):
    key = (device.index, stream)
    ws = _WORK.get(key)
    if ws is None:
        cap = sort_capacity(MAX_M)
        ws = {"q": torch.empty(cap, I // 2, dtype=torch.uint8, device=device),
              "qs": torch.empty(cap, I // 32, dtype=torch.uint8, device=device),
              "slots": torch.empty(cap, H, dtype=torch.bfloat16, device=device),
              "ready": torch.zeros(cap // BM * 64 + 64, dtype=torch.int32, device=device)}
        _WORK[key] = ws
    return ws


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _experts(x_codes: torch.Tensor, x_scales: torch.Tensor, weights: MoeWeights, sort: Sort,
             out: torch.Tensor | None = None, *, fused: bool = True, reduce: bool = True) -> torch.Tensor:
    """The routed experts of ``m`` tokens (M <= 192): ``x_codes`` u8/fp4x2 [m, 1792], ``x_scales`` e8m0/u8 [m, 112]
    (row-major MXFP4 of the bf16 activations, ``quantize_activations``), ``sort`` of the router's top-16 -> BF16
    [m, 3584]. Slot k of a token = router weight x down(SiTUv2(gate, up)) rounded once to BF16 in the sorted-row slot
    buffer; the output sums the 16 slots in slot order in fp32 and rounds once: no atomics, bitwise repeatable, graph
    replay = eager. GEMM1 and GEMM2 run in one launch (``fused``) or two; the sum is a third launch.
    ``reduce=False`` skips the sum and returns the slot buffer (bf16 [sort_capacity(192), 3584], one persistent workspace per
    device and stream, valid until the next call on that stream): row ``sort.reverse_sorted[t * 16 + k]`` is token t's slot k,
    already multiplied by its router weight; the caller sums k = 0..15 in fp32 and rounds once."""
    m = x_codes.shape[0]
    assert 1 <= m <= MAX_M and x_codes.shape[1] == H // 2 and x_scales.shape == (m, H // 32)
    dev = x_codes.device
    if out is None and reduce:
        out = torch.empty(m, H, dtype=torch.bfloat16, device=dev)
    assert not reduce or out.shape == (m, H) and out.dtype == torch.bfloat16 and out.is_contiguous()
    cap = sort_capacity(m)
    assert sort.sorted_token_ids.numel() >= cap and sort.sorted_expert_ids.numel() >= cap // BM
    assert sort.reverse_sorted.numel() >= m * TOPK
    stream = torch.cuda.current_stream()
    ws = _workspace(dev, stream.cuda_stream)
    xq, xs = x_codes.view(torch.uint8).view(-1), x_scales.view(torch.uint8).view(-1)
    ids, wt, eids, nv = sort.sorted_token_ids, sort.sorted_weights, sort.sorted_expert_ids, sort.num_valid_ids
    if fused:
        _run(compile_fused(), xq, xs, weights.w13.view(-1), weights.s13.view(-1), weights.w2.view(-1), weights.s2.view(-1),
             ws["q"].view(-1), ws["qs"].view(-1), ws["slots"].view(-1), ids, wt, eids, nv, m, cap,
             ws["ready"].data_ptr(), stream)
    else:
        _run(compile_gemm1(), xq, xs, weights.w13.view(-1), weights.s13.view(-1), ws["q"].view(-1), ws["qs"].view(-1),
             ids, eids, nv, m, cap, stream)
        _run(compile_gemm2(), ws["q"].view(-1), ws["qs"].view(-1), weights.w2.view(-1), weights.s2.view(-1),
             ws["slots"].view(-1), ids, wt, eids, nv, m, cap, stream)
    if not reduce:
        return ws["slots"]
    _run(compile_reduce(), ws["slots"].view(-1), out.view(-1), sort.reverse_sorted, m, cap, stream)
    return out


def _quantize(x: torch.Tensor, codes: torch.Tensor | None = None, scales: torch.Tensor | None = None):
    """MXFP4 of bf16 ``x`` [m, 3584]: (codes u8 [m, 1792], e8m0 scales u8 [m, 112]), aiter's rounding."""
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_contiguous()
    if codes is None:
        codes = torch.empty(m, H // 2, dtype=torch.uint8, device=x.device)
        scales = torch.empty(m, H // 32, dtype=torch.uint8, device=x.device)
    if m:
        _run(compile_quant(), x.view(-1), codes.view(-1), scales.view(-1), m, torch.cuda.current_stream())
    return codes, scales


def _mode(x_codes, x_scales, weights, sort, out=None, *, fused=True, reduce=True):
    return (bool(fused), bool(reduce))


def _inputs(mode, n, x_codes, x_scales, weights, sort, out=None, *, fused=True, reduce=True):
    dev = x_codes.device
    cap = sort_capacity(n)
    z = lambda k, dt: torch.zeros(k, dtype=dt, device=dev)
    s = Sort(z(cap, torch.int32), z(cap, torch.float32), z(cap // BM, torch.int32), z(2, torch.int32),
             z(n * TOPK, torch.int32))
    return (torch.zeros(n, H // 2, dtype=torch.uint8, device=dev), torch.zeros(n, H // 32, dtype=torch.uint8, device=dev),
            weights, s), {"fused": fused, "reduce": reduce}


mxfp4_moe_experts = entry("moe/mxfp4_h3584_i3072_e896_top16", mode=_mode, sizes=lambda mode: [MAX_M],
                          inputs=_inputs)(_experts)
quantize_activations = entry("moe/mxfp4_h3584_i3072_e896_top16.quant", mode=lambda x, *a, **k: (), sizes=lambda mode: [MAX_M],
                             inputs=lambda mode, n, x, *a, **k: ((torch.zeros(n, H, dtype=torch.bfloat16, device=x.device),), {}))(_quantize)


def mxfp4_moe(x: torch.Tensor, weights: MoeWeights, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
              *, fused: bool = True) -> torch.Tensor:
    """bf16 [m, 3584] -> routed experts [m, 3584] bf16: quantize, aiter sort, experts (tests and benchmarks)."""
    codes, scales = quantize_activations(x)
    return mxfp4_moe_experts(codes, scales, weights, moe_sort(topk_ids, topk_weights), fused=fused)
