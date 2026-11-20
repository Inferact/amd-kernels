# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 output as the o_proj's input (TP4: 1 KV head, N 6144 x K 2048; TP2: 2 KV heads, K 4096): at the served
decode shapes (8 rows a request = DSpark 7, 4 = DSpark 3; 1-64 requests; contexts 32K-192K) the epilogue writes, in the
code layout
``gemm.prequant_code_tile_rows`` names for the row count, the bytes the served quantizer makes of the bf16 output
(``gemm.mxfp8_linear``'s: the shuffled quantizer on the decode rows, the row-major one elsewhere), so the o_proj GEMM
on those codes equals the bf16 chain's GEMM bit for bit wherever both take the same table row (above 32 rows); graph
replay of attention + GEMM equals eager.
"""

import pytest
import torch
from amd_kernels.gemm.mxfp8_common import gemm as G

from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    GROUP,
    HEAD_DIM,
    aiter_scale_offsets,
    tile_code_offsets,
)

from .test_sparse_decode import make_case, require_gfx950, run

N = 6144  # the o_proj: K = 2048 per KV head
SERVED = [(1, 8, r) for r in (1, 2, 3, 4, 5, 6, 8, 13, 16, 32, 48, 64)] + [
    (1, 4, r) for r in (1, 4, 8, 9, 16, 32, 64)
] + [(2, 8, r) for r in (1, 2, 3, 4, 6, 8, 16, 32, 64)] + [(2, 4, r) for r in (1, 8, 16, 64)]


def long_lengths(reqs, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(32768, 196609, (reqs,), generator=gen).tolist()


def oproj_weight(K, seed=3):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn((N, K), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 124, (N, K // 32), generator=gen, device="cuda").to(torch.uint8)
    return G.preshuffle_mxfp8_weight(w, s)


def code_buffers(rows, tile, K, fill=0xA5):
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
    K = codes.shape[1]
    if tile:
        c = codes.view(-1)[tile_code_offsets(rows, K, codes.device)]
    else:
        c = codes[:rows]
    return c, scales.view(-1)[aiter_scale_offsets(rows, K, scales.device)]


@pytest.mark.parametrize("hkv,ql,reqs", SERVED, ids=[f"kv{h}-ql{a}-r{b}" for h, a, b in SERVED])
@torch.inference_mode()
def test_served_quantizer_bytes_and_oproj(hkv, ql, reqs):
    require_gfx950()
    case = make_case(hkv, ql, long_lengths(reqs, reqs * 10 + ql), seed=reqs + ql)
    rows = reqs * ql
    K = hkv * GROUP * HEAD_DIM
    assert case["q"].shape[0] == rows
    tile = G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    compute = torch.bfloat16 if hkv == 1 else torch.float16
    out = run(case, compute)
    codes, scales = code_buffers(rows, tile, K)
    run(case, compute, out=torch.empty_like(out), quant=(codes, scales, tile))
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
    # padding rows of the code tile are not written
    if tile and rows % max(tile, 16):
        pad = tile_code_offsets(codes.shape[0], K, codes.device)[rows:]
        assert (codes.view(-1)[pad] == 0xA5).all()
    w, ws = oproj_weight(K)
    got = gemm_codes(codes, scales, tile, rows, w, ws)
    ref = G.mxfp8_linear(out2d, w, ws, torch.bfloat16)
    torch.cuda.synchronize()
    same_row = G._select_config(rows, N, K, torch.bfloat16).kernel != G._DECODE_FQ_KERNEL
    if same_row:
        assert torch.equal(got, ref)
    else:  # the bf16 chain's fused-quant row sums K in another order
        rel = (got.float() - ref.float()).norm() / ref.float().norm()
        assert rel < 1e-2, float(rel)


GRAPH = [(1, 8, 1), (1, 8, 4), (1, 8, 6), (1, 8, 16), (1, 8, 64), (1, 4, 32), (1, 4, 64), (2, 8, 1), (2, 8, 4),
         (2, 8, 16), (2, 4, 32)]


@pytest.mark.parametrize("hkv,ql,reqs", GRAPH, ids=[f"kv{h}-ql{a}-r{b}" for h, a, b in GRAPH])
@torch.inference_mode()
def test_graph_replay_attention_and_oproj(hkv, ql, reqs):
    """attention (MXFP8 out) + o_proj captured: every replay writes the eager codes,
    scales and o_proj output, also after new q / cache values are copied in."""
    require_gfx950()
    lengths = long_lengths(reqs, reqs * 3 + ql)
    a, b = make_case(hkv, ql, lengths, seed=5), make_case(hkv, ql, lengths, seed=5)
    b["q"] = (torch.randn_like(a["q"].float()) * 1.5).bfloat16()
    b["kv_cache"] = torch.randint_like(a["kv_cache"], 0, 256)
    for side in range(2):  # random codes, finite block scales (test_graph_replay's)
        start = side * hkv * 9216 + hkv * 8192
        sc_b = b["kv_cache"].view(b["kv_cache"].shape[0], -1)[:, start : start + hkv * 1024]
        sc_b.masked_fill_((sc_b & 0x7F) == 0x7F, 0x30)
    rows = reqs * ql
    K = hkv * GROUP * HEAD_DIM
    tile = G.prequant_code_tile_rows(rows, N, K, torch.bfloat16)
    w, ws = oproj_weight(K)
    compute = torch.bfloat16 if hkv == 1 else torch.float16

    def chain(case, codes, scales):
        run(case, compute, out=torch.empty_like(case["q"]), quant=(codes, scales, tile))
        return gemm_codes(codes, scales, tile, rows, w, ws)

    def eager(case):
        codes, scales = code_buffers(rows, tile, K)
        o = chain(case, codes, scales)
        torch.cuda.synchronize()
        return real_bytes(codes, scales, rows, tile), o.clone()

    want_a, want_b = eager(a), eager(b)
    g_codes, g_scales = code_buffers(rows, tile, K)
    chain(a, g_codes, g_scales)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out = chain(a, g_codes, g_scales)
    for want, src in ((want_a, None), (want_a, None), (want_b, b)):
        if src is not None:
            a["q"].copy_(src["q"])
            a["kv_cache"].copy_(src["kv_cache"])
        g_codes.fill_(0xA5)
        g_scales.fill_(0xA5)
        g_out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        (wc, wsc), wo = want
        gc, gs = real_bytes(g_codes, g_scales, rows, tile)
        assert torch.equal(gc, wc) and torch.equal(gs, wsc)
        assert torch.equal(g_out, wo)
