# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared test and benchmark inputs: MXFP4 routed experts in the chain's layouts, routing, activations, replay."""

import pytest
import torch
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.layout import w2_layout, w13_layout
from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.prefill_h6144_i768 import H, I, Mxfp4PrefillScales

E, TOPK = 128, 4  # the served shape: 128 routed experts, top-4 (the shared slot runs elsewhere)
ROUTED_SCALING = 2.0  # MiniMax-M3.1; the top-k kernel folds it into the weights

# vLLM's W4A4 call sizes (W4A4_FROM_TOKENS 1536 up to max_num_batched_tokens 65536), some off the 32 / 256-row grid
SERVED_TOKENS = (1536, 2049, 4097, 16384, 32768, 65536)


def require_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        pytest.skip(f"gfx950 only, found {arch}")


def make_experts(num_experts: int = E, seed: int = 0, cpu_gen: bool = False):
    """Random MXFP4 experts, scales over 6 binades (the checkpoint's span <= 7): (raw, (w13, w2, scales)), ``raw`` =
    (w13 codes, w13 e8m0, w2 codes, w2 e8m0) row-major. ``cpu_gen``: vLLM's test's CPU generator (its exact tensors)."""
    if cpu_gen:
        g = torch.Generator().manual_seed(seed)

        def rnd(lo, hi, shape):
            return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=g).cuda()

        w13, s13 = rnd(0, 256, (num_experts, 2 * I, H // 2)), rnd(118, 124, (num_experts, 2 * I, H // 32))
        w2, s2 = rnd(0, 256, (num_experts, H, I // 2)), rnd(118, 124, (num_experts, H, I // 32))
    else:
        g = torch.Generator(device="cuda").manual_seed(seed)

        def rnd(lo, hi, shape):
            return torch.randint(lo, hi, shape, dtype=torch.uint8, device="cuda", generator=g)

        w13, s13 = rnd(0, 256, (num_experts, 2 * I, H // 2)), rnd(118, 124, (num_experts, 2 * I, H // 32))
        w2, s2 = rnd(0, 256, (num_experts, H, I // 2)), rnd(118, 124, (num_experts, H, I // 32))
    ops = (w13_layout(w13), w2_layout(w2), Mxfp4PrefillScales.from_row_major(s13, s2))
    return (w13, s13, w2, s2), ops


def make_shared(seed: int = 607):
    """Random MXFP8 shared-expert weights: (raw, SharedWeights), ``raw`` = (w13 e4m3 [1536, 6144] u8, e8m0 [1536, 192],
    w2 e4m3 [6144, 768] u8, e8m0 [6144, 24]) row-major."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import SharedWeights
    g = torch.Generator(device="cuda").manual_seed(seed)

    def weight(n, k):
        q = (torch.randn((n, k), device="cuda", generator=g) * 20).to(torch.float8_e4m3fn).view(torch.uint8)
        s = torch.randint(118, 124, (n, k // 32), device="cuda", dtype=torch.uint8, generator=g)
        return q, s

    raw = (*weight(2 * I, H), *weight(H, I))
    return raw, SharedWeights.from_row_major(*raw)


def small_slots(x, w, topk_weights, topk_ids, **kwargs):
    """The small-tile host as the BF16 entry runs it to 448 rows (shared tiles folded in, ``id_rows`` at its rows):
    (slots [M, 4, H], shared output [M, H]) of ``MoeWeights`` ``w`` on the BF16 ``x`` and the given routing."""
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.experts_a4w4_small import a4w4_experts, id_rows_for
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.quant import mxfp4_quant
    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4.shared_quant import shared_quant
    kwargs.setdefault("id_rows", id_rows_for(x.shape[0]))
    return a4w4_experts(mxfp4_quant(x), w.w13, w.w2, w.a4w4_scales, topk_weights, topk_ids,
                        shared=(*shared_quant(x), w.shared_weights), **kwargs)


def make_routing(tokens: int, num_experts: int = E, seed: int = 11, skewed: bool = False):
    """(fp32 weights [M, 4], int32 ids [M, 4]) as the top-k kernel hands the chain: distinct experts per token, weights
    renormalized times the routed scaling. ``skewed``: expert probabilities ~ rank^-1.2 (some experts get no row)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    if skewed:
        p = torch.arange(1, num_experts + 1, device="cuda", dtype=torch.float32) ** -1.2
        ids = torch.multinomial(p.expand(tokens, num_experts), TOPK, generator=g)
    else:
        ids = torch.rand((tokens, num_experts), device="cuda", generator=g).topk(TOPK).indices
    w = torch.rand((tokens, TOPK), device="cuda", generator=g) + 0.5
    w = w / w.sum(1, keepdim=True) * ROUTED_SCALING
    return w.contiguous(), ids.int().contiguous()


def make_activation(tokens: int, seed: int = 13, scale: float = 0.5):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn((tokens, H), device="cuda", generator=g) * scale).bfloat16()


def make_mxfp8(tokens: int, seed: int = 17):
    """A gathered MXFP8 input: (e4m3 codes [M, H] u8 without NaN, e8m0 [M, H/32])."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (tokens, H), dtype=torch.uint8, device="cuda", generator=g)
    codes[(codes & 0x7F) == 0x7F] = 0  # e4m3 NaN
    scales = torch.randint(110, 145, (tokens, H // 32), dtype=torch.uint8, device="cuda", generator=g)
    return codes, scales


def two_launch_slots(x_quant, w, topk_weights, topk_ids):
    """The slots [M, 4, H] of the BM256 host's two launches at any M (``experts_a4w4`` above ``GEMM12_TOKENS``: the
    persistent full-height GEMM1, then a GEMM2 that stores every valid row): the bitwise oracle of the one-launch and
    short-tile kernels."""
    from unittest import mock

    from amd_kernels.moe.mxfp4_h6144_i768_e128_top4 import experts_a4w4
    with mock.patch.object(experts_a4w4, "GEMM12_TOKENS", (0, 0)):
        return experts_a4w4.a4w4_experts(x_quant, w.w13, w.w2, w.a4w4_scales, topk_weights, topk_ids)


def tile_map_ref(sorted_expert_ids, num_valid_ids, grid, bm):
    """GEMM1's tile map (``common/tile_map.py``) of a routing sort's block experts, in torch: each expert's run of
    m-blocks n-major, -1 to ``grid``, the valid entry count last."""
    nb_n = I // 128
    n_bits = (nb_n - 1).bit_length()
    vb = int(num_valid_ids[0]) // bm
    eids = sorted_expert_ids[:vb].tolist()
    tm = [-1] * (grid + 1)
    lo = 0
    while lo < vb:
        hi = lo
        while hi < vb and eids[hi] == eids[lo]:
            hi += 1
        for m in range(lo, hi):
            for n in range(nb_n):
                tm[nb_n * lo + n * (hi - lo) + (m - lo)] = m << n_bits | n
        lo = hi
    tm[grid] = vb * nb_n
    return torch.tensor(tm, dtype=torch.int32, device="cuda")


def replay_check(fn, outputs_of=lambda out: out):
    """Run ``fn`` eagerly, capture it in a HIP graph, poison the graph's outputs (NaN for floats, 0xA5 bytes otherwise),
    replay, and return the (eager, replayed) output lists."""
    eager = [t.clone() for t in outputs_of(fn())]
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = outputs_of(fn())
    for t in captured:
        if t.dtype.is_floating_point and t.element_size() > 1:
            t.fill_(float("nan"))
        else:
            t.view(torch.uint8).fill_(0xA5)
    graph.replay()
    torch.cuda.synchronize()
    return eager, captured
