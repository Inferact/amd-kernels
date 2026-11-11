# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 output as the TP4 DSpark draft's o_proj input (vLLM with the library's draft
path and GEMMs): at the served draft shapes (16 / 1 heads, 128-token pages,
window 4096, 7 rows a request = DSpark 7, 3 = DSpark 3, non-causal as the served draft
and causal; 1-64 requests) the epilogue writes, in the code layout
``gemm.prequant_code_tile_rows`` names for the row count, the bytes the served quantizer
(``gemm.mxfp8_linear``'s activation quantizer) makes of the bf16 output, so the
o_proj GEMM on those codes equals the bf16 chain's GEMM bit for bit wherever both take the
same table row (above 32 rows); graph replay of attention + GEMM equals eager.
"""

import pytest
import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    aiter_scale_offsets,
    tile_code_offsets,
)
from amd_kernels.gemm.mxfp8_common import gemm as G

from .test_decode import launch, make_batch, require_gfx950

N, K = 6144, 2048  # the TP4 o_proj
# 3 rows (DSpark 3) take the generic kernel (kernel.py): 6 splits at 1-40 requests, 5 at
# 41-48, 4 at 49-64. Its non-causal MXFP8 code objects with 3, 5 or 6 splits read row 0's
# P right after the v_exp_f32 writing it; an inline-asm bf16 pack there gets no wait state
# (NaN codes for row 0, heads 0-3 and 8-11), the native fptrunc does.
# test_split_counts_ql3 walks every split count.
SERVED = [
    pytest.param(ql, r, causal, id=f"ql{ql}-r{r}-{'c' if causal else 'nc'}")
    for ql, reqs in ((7, (1, 2, 4, 5, 6, 7, 8, 11, 16, 32, 48, 49, 64)), (3, (1, 11, 16, 32, 41, 48, 64)))
    for r in reqs
    for causal in (False, True)
]


def oproj_weight(seed=3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((N, K), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 124, (N, K // 32), generator=gen, device="cuda").to(torch.uint8)
    return G.preshuffle_mxfp8_weight(w, s)


def code_buffers(rows, tile, fill=0xA5):
    pad = max(tile, 16)
    rows_p = (rows + pad - 1) // pad * pad
    codes = torch.full((rows_p, K), fill, dtype=torch.uint8, device="cuda")
    scales = torch.full(((rows_p + 31) // 32 * 32, K // 32), fill, dtype=torch.uint8, device="cuda")
    return codes, scales


def gemm_codes(codes, scales, tile, rows, w, ws):
    c8 = codes.view(torch.float8_e4m3fn)
    if tile:
        return G.mxfp8_gemm_shuffled(c8, scales, w, ws, torch.bfloat16, rows)
    return G.mxfp8_gemm(c8[:rows], scales, w, ws, torch.bfloat16, activation_scale_preshuffled=True)


def real_bytes(codes, scales, rows, tile):
    if tile:
        c = codes.view(-1)[tile_code_offsets(rows, K, codes.device)]
    else:
        c = codes[:rows]
    return c, scales.view(-1)[aiter_scale_offsets(rows, K, scales.device)]


def check_served_bytes(batch, ql, causal, **kw):
    """The epilogue's codes / scales equal the served quantizer of the bf16 output of
    the same launch configuration (``kw``: explicit splits / XCD grid)."""
    q = batch[4]
    rows = q.shape[0]
    tile = G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    out = torch.full_like(q, float("nan"))
    launch(batch, ql, causal, out, **kw)
    codes, scales = code_buffers(rows, tile)
    launch(batch, ql, causal, None, (codes, scales, tile), **kw)
    torch.cuda.synchronize()
    out2d = out.view(rows, K)
    if tile:
        want_c, want_s = G.mxfp8_e4m3_quantize_shuffled(out2d, tile)
    else:
        want_c, want_s = G.mxfp8_e4m3_quantize_aiter(out2d)
    got_c, got_s = real_bytes(codes, scales, rows, tile)
    ref_c, ref_s = real_bytes(want_c.view(torch.uint8), want_s.view(torch.uint8), rows, tile)
    assert torch.equal(got_c, ref_c), int((got_c != ref_c).sum())
    assert torch.equal(got_s, ref_s), int((got_s != ref_s).sum())
    return out2d, codes, scales, tile


@pytest.mark.parametrize("causal", [False, True], ids=["nc", "c"])
@pytest.mark.parametrize("xcd", [None, 1], ids=["xcd", "noxcd"])
@pytest.mark.parametrize("splits", [1, 2, 3, 4, 5, 6])
@torch.inference_mode()
def test_split_counts_ql3(splits, xcd, causal):
    """3 rows, 1 request, every split count the plan can take, with and without the
    XCD grid (3, 5 and 6 splits are the ones exposed to the v_exp_f32 -> pack hazard)."""
    require_gfx950()
    gen = torch.Generator().manual_seed(13)
    batch = make_batch(1, 3, 0, gen, 128, 1)
    check_served_bytes(batch, 3, causal, splits=splits, xcd_count=xcd)


@pytest.mark.parametrize("ql,reqs,causal", SERVED)
@torch.inference_mode()
def test_served_quantizer_bytes_and_oproj(ql, reqs, causal):
    require_gfx950()
    gen = torch.Generator().manual_seed(reqs * 10 + ql)
    batch = make_batch(reqs, ql, 0, gen, 128, 1)
    rows = batch[4].shape[0]
    out2d, codes, scales, tile = check_served_bytes(batch, ql, causal)
    w, ws = oproj_weight()
    got = gemm_codes(codes, scales, tile, rows, w, ws)
    ref = G.mxfp8_linear(out2d, w, ws, torch.bfloat16)
    torch.cuda.synchronize()
    if G._select_config(rows, N, K, torch.bfloat16).kernel != G._DECODE_FQ_KERNEL:
        assert torch.equal(got, ref)
    else:  # the bf16 chain's fused-quant row sums K in another order
        rel = (got.float() - ref.float()).norm() / ref.float().norm()
        assert rel < 1e-2, float(rel)


GRAPH = [(7, 1), (7, 6), (7, 7), (7, 16), (7, 49), (7, 64), (3, 16), (3, 48), (3, 64)]


@pytest.mark.parametrize("ql,reqs", GRAPH, ids=[f"ql{a}-r{b}" for a, b in GRAPH])
@torch.inference_mode()
def test_graph_replay_attention_and_oproj(ql, reqs):
    """attention (MXFP8 out, non-causal) + o_proj captured: every replay writes the
    eager codes, scales and o_proj output, also after new q values are copied in."""
    require_gfx950()
    gen = torch.Generator().manual_seed(reqs * 3 + ql)
    batch = make_batch(reqs, ql, 0, gen, 128, 1)
    q = batch[4]
    rows = q.shape[0]
    tile = G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    w, ws = oproj_weight()
    q_a, q_b = q.clone(), (torch.randn_like(q.float()) * 1.5).bfloat16()

    def chain(codes, scales):
        launch(batch, ql, False, None, (codes, scales, tile))
        return gemm_codes(codes, scales, tile, rows, w, ws)

    wants = []
    for src in (q_a, q_b):
        q.copy_(src)
        codes, scales = code_buffers(rows, tile)
        o = chain(codes, scales)
        torch.cuda.synchronize()
        wants.append((real_bytes(codes, scales, rows, tile), o.clone()))
    q.copy_(q_a)
    g_codes, g_scales = code_buffers(rows, tile)
    chain(g_codes, g_scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = chain(g_codes, g_scales)
    for want, src in ((wants[0], q_a), (wants[0], q_a), (wants[1], q_b)):
        q.copy_(src)
        g_codes.fill_(0xA5)
        g_scales.fill_(0xA5)
        g_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        (wc, wsc), wo = want
        gc, gs = real_bytes(g_codes, g_scales, rows, tile)
        assert torch.equal(gc, wc) and torch.equal(gs, wsc)
        assert torch.equal(g_out, wo)
