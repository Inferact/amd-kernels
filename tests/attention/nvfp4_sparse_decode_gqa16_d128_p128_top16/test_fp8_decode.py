# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The fp8-MFMA decode (``fp8/kernel.py`` per row, ``fp8/row_quad.py`` four-row union;
``compute_type=torch.float8_e4m3fn``): NV's Q8KV4 numerics with P rounded against its chunk's own max (32 keys of a
block), so a row's P bits do not depend on splits, unions or M (``reference.py``, "The fp8 kernels' P rule").

Contract: ``reference.fp8_chunk_decode_reference`` within the bf16 output's rounding; NV's slot-order reference only
within the rule's order effect (rel. L2 3.4e-2 on these inputs between the slot-order reference and itself with the
slots reversed). Cross-M: the same request alone and inside 32 / 128 requests (per-row splits 4 / 1, the union) agrees
to cos 0.99999 per (row, head)."""

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    aiter_scale_offsets,
    cosine,
    fp8_chunk_decode_reference,
    mxfp8_quantize,
    nv_sparse_decode_reference,
    rel_l2,
    tile_code_offsets,
)

from .test_sparse_decode import (
    SM_SCALE,
    lengths_for,
    live_rows,
    make_case,
    quant_buffers,
    require_gfx950,
    run,
)

FP8 = torch.float8_e4m3fn
ORDER_REL_L2 = 3e-3  # bf16 output rounding (~1.8e-3) + rare flipped P roundings
NV_REL_L2 = 5e-2  # the rule's order effect (module docstring)


def unit_scales(case):
    case["k_scale"] = torch.ones((), dtype=torch.float32, device="cuda")
    case["v_scale"] = torch.ones((), dtype=torch.float32, device="cuda")
    return case


@pytest.mark.parametrize(
    "hkv,ql,reqs,splits,xcd",
    [(1, 8, 1, 4, 8), (1, 8, 5, 4, 8), (1, 8, 9, 2, 8), (1, 8, 24, 1, 8), (1, 8, 40, 2, 8), (1, 8, 4, 8, 0),
     (1, 4, 6, 3, 8)],
)
def test_contract_fp8(hkv, ql, reqs, splits, xcd):
    require_gfx950()
    case = unit_scales(make_case(hkv, ql, lengths_for(reqs, seed=reqs), seed=reqs))
    out = run(case, FP8, splits=splits, xcd_count=xcd)
    live = live_rows(case)
    assert torch.isfinite(out).all()
    args = (case["q"], case["kv_cache"], case["topk_idx"], case["block_table"], case["seq_lens"],
            SM_SCALE, ql)
    ref = fp8_chunk_decode_reference(*args)
    rel, cos = rel_l2(out[live], ref[live]), cosine(out[live], ref[live])
    nv = nv_sparse_decode_reference(*args).double()
    rel_nv = rel_l2(out[live], nv[live])
    print(f"fp8 decode vs the chunk reference: rel L2 {rel:.2e}, cos {cos:.7f}; vs NV {rel_nv:.2e}")
    assert rel < ORDER_REL_L2 and cos > 0.99999, (rel, cos)
    assert rel_nv < NV_REL_L2, rel_nv
    assert (out[~live] == 0).all()


def test_global_scales_fp8():
    """k_scale folds into the score scale, v_scale into the output (as the bf16 path)."""
    require_gfx950()
    case = make_case(1, 8, lengths_for(6, seed=3), seed=3)
    case["k_scale"] = torch.tensor(0.5, dtype=torch.float32, device="cuda")
    case["v_scale"] = torch.tensor(2.0, dtype=torch.float32, device="cuda")
    out = run(case, FP8, splits=2)
    ref = fp8_chunk_decode_reference(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"], case["seq_lens"],
        SM_SCALE, 8, k_scale=0.5, v_scale=2.0,
    )
    live = live_rows(case)
    assert rel_l2(out[live], ref[live]) < ORDER_REL_L2


@pytest.mark.parametrize("reqs", [2, 5, 16, 20, 32, 33, 40, 41, 48, 65, 72, 73, 80])
def test_fp8_bitwise_graph_and_quant(reqs):
    """Three launches bit for bit; a CUDA graph replay equals the eager launch; the MXFP8 output's bytes equal
    mxfp8_quantize of the same schedule's bf16 output; padding requests and rows past the batch untouched."""
    require_gfx950()
    case = unit_scales(make_case(1, 8, lengths_for(reqs, seed=11, pad=1), seed=11))
    a = run(case, FP8)
    for _ in range(2):
        assert torch.equal(run(case, FP8), a)
    out = torch.full_like(case["q"], float("nan"))
    run(case, FP8, out=out)  # warm-up (compile) outside the capture
    g = torch.cuda.CUDAGraph()
    out.fill_(float("nan"))
    with torch.cuda.graph(g):
        run(case, FP8, out=out)
    out.fill_(float("nan"))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, a)
    rows, width = case["q"].shape[0], case["q"].shape[1] * case["q"].shape[2]
    for tile in (0, 16):
        codes, sc = quant_buffers(rows, width)
        run(case, FP8, quant=(codes, sc, tile))
        want_c, want_s = mxfp8_quantize(a.view(rows, width).float())
        got_c = codes[:rows] if not tile else codes.view(-1)[tile_code_offsets(rows, width, codes.device)]
        got_s = sc.view(-1)[aiter_scale_offsets(rows, width, sc.device)]
        live = live_rows(case)
        assert torch.equal(got_c.view(rows, -1)[live], want_c.view(rows, -1)[live]), tile
        assert torch.equal(got_s.view(rows, -1)[live], want_s.view(rows, -1)[live]), tile


def test_fp8_random_bytes_past_lengths():
    """Random cache bytes (NaN-free scales) and tail slots: finite, and the contract."""
    require_gfx950()
    case = unit_scales(make_case(1, 8, lengths_for(5, seed=21), seed=21, random_bytes=True))
    out = run(case, FP8, splits=4)
    assert torch.isfinite(out).all()
    ref = fp8_chunk_decode_reference(
        case["q"], case["kv_cache"], case["topk_idx"], case["block_table"], case["seq_lens"],
        SM_SCALE, 8,
    )
    live = live_rows(case)
    assert rel_l2(out[live], ref[live]) < ORDER_REL_L2


@pytest.mark.parametrize("reqs", [97, 128, 129, 256])
def test_fp8_union_graph_quant_and_contract(reqs):
    """The union path: padded requests, a partial 16-row code tile, graph, determinism, both output layouts, and the
    contract on sampled rows."""
    require_gfx950()
    case = unit_scales(make_case(1, 8, lengths_for(reqs, seed=23, pad=1), seed=23))
    out = run(case, FP8)
    assert torch.isfinite(out).all()
    assert (out[~live_rows(case)] == 0).all()
    for _ in range(2):
        assert torch.equal(run(case, FP8), out)
    rows = sorted({0, 1, 2, 3, 4, 5, 7, 8, out.shape[0]//2, out.shape[0]-1})
    ref = fp8_chunk_decode_reference(case['q'], case['kv_cache'], case['topk_idx'],
        case['block_table'], case['seq_lens'], SM_SCALE, 8, rows=rows)
    err = rel_l2(out[rows], ref)
    print('FP8 union vs the chunk reference: rel L2', err)
    assert err < ORDER_REL_L2
    assert cosine(out[rows],ref) > 0.99999
    for tile in (0,16):
        codes,sc = quant_buffers(out.shape[0],16*128)
        run(case,FP8,quant=(codes,sc,tile))
        want_c,want_s=mxfp8_quantize(out.view(out.shape[0],-1).float())
        got_c=codes[:out.shape[0]] if not tile else codes.flatten()[tile_code_offsets(out.shape[0],16*128,codes.device)].reshape_as(want_c)
        got_s=sc.flatten()[aiter_scale_offsets(out.shape[0],16*128,sc.device)].reshape_as(want_s)
        assert torch.equal(got_c,want_c)
        assert torch.equal(got_s,want_s)
    graph=torch.cuda.CUDAGraph()
    target=torch.empty_like(out)
    run(case,FP8,out=target)
    with torch.cuda.graph(graph): run(case,FP8,out=target)
    target.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
    assert torch.equal(target,out)


@pytest.mark.parametrize("reqs", [1, 2, 4, 5, 16, 20, 32, 40, 48, 64, 65, 72, 80, 81, 128])
def test_fp8_strided_poison_and_mfma(reqs):
    from .test_sparse_decode import poison_tail
    from tests.mfma_spinner import mfma_spinner
    require_gfx950()
    case=unit_scales(make_case(1,8,lengths_for(reqs,seed=29,pad=1 if reqs > 1 else 0),seed=29))
    clean=run(case,FP8)
    poison_tail(case,1)
    dirty=run(case,FP8)
    assert torch.equal(clean,dirty)
    cache=case['kv_cache'];size=cache[0].numel()
    wide=torch.full((cache.shape[0],size*2),0x7f,dtype=torch.uint8,device='cuda')
    view=wide[:,size:].view(cache.shape);view.copy_(cache);case['kv_cache']=view
    assert torch.equal(run(case,FP8),clean)
    main,side=torch.cuda.Stream(),torch.cuda.Stream(priority=-1)
    spin=mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000,side)
        with torch.cuda.stream(main): got=run(case,FP8)
        torch.cuda.synchronize()
        assert torch.equal(got,clean)


@pytest.mark.parametrize("many", [4, 32, 33, 40, 41, 48, 65, 72, 73, 80, 81, 96, 97, 128])
def test_fp8_cross_m_cosine_gate(many):
    """The cross-M gate: one request alone (4 splits) and the same request inside ``many`` copies (4 / 2 / 1 splits
    per row, the union from 97) agree to cos 0.99999 per (row, head): only fp32 summation orders differ."""
    require_gfx950()
    case = unit_scales(make_case(1, 8, [4097], seed=23))
    small = run(case, FP8)
    big = dict(case)
    big["q"] = case["q"].repeat(many, 1, 1)
    big["topk_idx"] = case["topk_idx"].repeat(1, many, 1)
    big["block_table"] = case["block_table"].repeat(many, 1)
    big["seq_lens"] = case["seq_lens"].repeat(many)
    large = run(big, FP8)
    cos = torch.nn.functional.cosine_similarity(
        small.double(), large[:8].double(), dim=-1
    ).min().item()
    tail = torch.nn.functional.cosine_similarity(
        small.double(), large[-8:].double(), dim=-1
    ).min().item()
    print(f"cross-M FP8 ({many} requests) min head cosine {cos:.8f} / last {tail:.8f}")
    assert min(cos, tail) >= 0.99999
