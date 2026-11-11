# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The FlyDSL NVFP4 sparse prefills (bf16 and fp16 compute; fp8 below) against the
torch contract in ``reference.py``, on the served shape: Hkv = 1 (TP4, 16 query heads),
1 to 65536 prefill tokens."""

import dataclasses

import pytest
import torch

from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import host, reference
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16 import nvfp4_sparse_prefill as prefill
from amd_kernels.attention.nvfp4_sparse_prefill_gqa16_d128_p128_top16.query import pack_query
from .cases import (
    BLOCK,
    HEAD_DIM,
    TOPK,
    TP4_HKV,
    make_case,
    require_gfx950,
)

VARIANTS = ["bf16", "fp16", "fp8"]  # fp16: the default compute dtype
HKVS = [TP4_HKV]
FILL = 0xA5


def run(case, variant, q=None, out=None, **kw):
    q = case.q if q is None else q
    out = torch.empty(q.shape, dtype=torch.bfloat16, device="cuda") if out is None else out
    prefill(
        *case.args(q),
        out,
        k_scale=case.k_scale,
        v_scale=case.v_scale,
        compute_dtype=variant,
        **kw,
    )
    return out


def run_quant(case, variant, n_tok=None, codes=None, scales=None, **kw):
    """The quant epilogue's codes and (AITER-shuffled) scales for the first ``n_tok``
    tokens, into fresh 0xA5-filled buffers unless given."""
    t = case.q.shape[0] if n_tok is None else n_tok
    k = case.q.shape[1] * HEAD_DIM
    padded = (case.q.shape[0] + 31) // 32 * 32
    if codes is None:
        codes = torch.full((padded, k), FILL, dtype=torch.uint8, device="cuda")
        scales = torch.full((padded, k // 32), FILL, dtype=torch.uint8, device="cuda")
    sub = case_slice(case, 0, t)
    untouched = torch.full(sub.q.shape, 17.0, dtype=torch.bfloat16, device="cuda")
    run(sub, variant, out=untouched, quant_out=(codes, scales), **kw)
    assert bool((untouched == 17).all()), "quant_out must leave out untouched"
    return codes, scales


def run_pack(case, variant, extra_rows=37, **kw):
    t, h = case.q.shape[:2]
    pack = torch.full((t + extra_rows, h * 132), FILL, dtype=torch.uint8, device="cuda")
    untouched = torch.full(case.q.shape, 17.0, dtype=torch.bfloat16, device="cuda")
    run(case, variant, out=untouched, pack_out=pack, **kw)
    assert bool((untouched == 17).all()), "pack_out must leave out untouched"
    return pack


def case_slice(case, a, b):
    """Tokens a..b of ``case`` as their own call (the same cache and table)."""
    if (a, b) == (0, case.q.shape[0]):
        return case
    return dataclasses.replace(
        case,
        q=case.q[a:b],
        topk=case.topk[:, a:b].contiguous(),
        req=case.req[a:b],
        pos=case.pos[a:b],
    )


def check_reference(case, out, variant, rows=None, fp8=False):
    rows = case.rows() if rows is None else rows
    ref = reference.sparse_prefill_reference(
        *case.args(),
        k_scale=case.k_scale,
        v_scale=case.v_scale,
        rows=rows,
        fp8=fp8 or variant == "fp8",
    )
    got = out[rows]
    assert torch.isfinite(got).all()
    err, cos = reference.rel_l2(got, ref), reference.cosine(got, ref)
    print(f"[{variant} hkv={case.hkv} T={case.q.shape[0]}] rel_l2={err:.3e} cos={cos:.8f}")
    assert err < reference.REL_L2_TOL[variant], (err, cos)
    assert cos > reference.COS_MIN[variant], (err, cos)
    return err


def check_quant_bytes(out, codes, scales, n_tok):
    """``quant_out`` = reference MXFP8 quant of ``out`` (rows < n_tok), rest 0xA5."""
    ref_codes, ref_scales = reference.mxfp8_quantize(out[:n_tok].reshape(n_tok, -1))
    assert torch.equal(codes[:n_tok], ref_codes)
    plain = reference.aiter_unshuffle_scales(scales)
    assert torch.equal(plain[:n_tok], ref_scales)


# --- the contract --------------------------------------------------------------------


@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 2.0)], ids=["unit", "k0.5v2"])
@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_matches_reference(variant, hkv, scales):
    """A batch with prefix context, 4-token tiles across requests and a request of 3
    tokens; global scales 1 / 1 and 0.5 / 2."""
    require_gfx950()
    case = make_case([1027, 514, 3, 262], [3000, 514, 1100, 900], hkv, scales, seed=1)
    check_reference(case, run(case, variant), variant)


@pytest.mark.parametrize("n_tok", [1, 3, 5])
@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_few_tokens(variant, hkv, n_tok):
    require_gfx950()
    case = make_case([n_tok], [700], hkv, seed=2)
    check_reference(case, run(case, variant), variant, rows=list(range(n_tok)))


@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_shared_picks_deterministic(variant, hkv):
    """Indexer-like picks (many pages shared by a tile's tokens), tiles across
    requests: the reference, and three launches bitwise equal (bf16 and packed)."""
    require_gfx950()
    case = make_case([1501, 6, 2050], [9000, 2100, 6200], hkv, seed=3, picks="shared")
    out = run(case, variant)
    check_reference(case, out, variant)
    for _ in range(2):
        assert torch.equal(run(case, variant), out)
    packs = [run_pack(case, variant) for _ in range(3)]
    assert torch.equal(packs[0], packs[1]) and torch.equal(packs[0], packs[2])


@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_one_and_two_round_combine_consistent(variant, hkv):
    """A fresh prompt (positions from 0) in one call (two-round combine) and as calls
    on both sides of the rounds boundary (more workgroups than CUs or not), cut at tile
    edges, preserves the cross-M contract. Small BF16 tiles and FP8 may change
    page order."""
    require_gfx950()
    cus = torch.cuda.get_device_properties(0).multi_processor_count
    edge = cus * 4 // hkv  # the most tokens with one workgroup per CU (rounds 1)
    n = 4 * edge
    case = make_case([n], [n], hkv, seed=5, picks="shared")
    full = run(case, variant)
    check_reference(case, full, variant)
    # BF16 / FP16 at one KV head: TQ1 (256), TQ2 (256, and 1536 = its largest), TQ4 (2048)
    small = variant in ("bf16", "fp16") and hkv == TP4_HKV
    cuts = [0, 256, 512, 2048, n] if small else [0, edge, 3 * edge, n]
    parts = torch.cat([run(case_slice(case, a, b), variant) for a, b in zip(cuts, cuts[1:])])
    if variant == "fp8" or small:
        # Small query tiles walk their local pages; the four-query layout
        # can visit the same pages in another order.
        # FP8 retains chunk-P rounding. BF16 / FP16 retain RNE P rounding, but
        # a different page order can change the running max seen by that rounding.
        a, b = full.double(), parts.double()
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).min().item()
        err = reference.rel_l2(full, parts)
        print(f"{variant} across layouts hkv={hkv}: min head cos {cos:.9f}, rel L2 {err:.4g}")
        assert cos >= 0.99999 and err <= 2e-3
    else:
        assert torch.equal(full, parts)


@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_block_outer_pages_bitwise(variant, hkv):
    """Pages as one layer's slice of wider blocks (BLHNC: stride(0) spans neighbouring
    layers' pages, filled with NaN scale bytes) give the same bits as a dense cache."""
    require_gfx950()
    case = make_case([1027, 3, 262], [3000, 1100, 900], hkv, seed=3)
    page = case.cache[0].numel()
    wide = torch.full((case.cache.shape[0], 3 * page), 0x7F, dtype=torch.uint8, device="cuda")
    view = wide[:, page : 2 * page].view(case.cache.shape)
    view.copy_(case.cache)
    assert view.stride(0) == 3 * case.cache.stride(0)
    out = run(case, variant)
    assert torch.equal(run(dataclasses.replace(case, cache=view), variant), out)


# --- MXFP8 epilogues -----------------------------------------------------------------


@pytest.mark.parametrize("q_lens", [[129, 67, 35], [1025]], ids=["small", "1025"])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 2.0)], ids=["unit", "k0.5v2"])
@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_quant_epilogue_bytes(variant, hkv, scales, q_lens):
    """``quant_out`` (o_proj's row-major codes + AITER scales) is the reference quant
    of the kernel's bf16 output byte for byte, padding rows and ``out`` untouched;
    reusing the buffers after shrinking T (across the rounds boundary) rewrites only
    the new rows."""
    require_gfx950()
    ctx = [max(c, q + 1024) for c, q in zip([4000, 2100, 1300], q_lens)]
    case = make_case(q_lens, ctx, hkv, scales, seed=19)
    n_tok = case.q.shape[0]
    out = run(case, variant)
    codes, sc = run_quant(case, variant)
    check_quant_bytes(out, codes, sc, n_tok)
    plain = reference.aiter_unshuffle_scales(sc)
    assert bool((plain[n_tok:] == FILL).all()) and bool((codes[n_tok:] == FILL).all())
    shorter = n_tok - 17
    before_codes, before_sc = codes.clone(), plain.clone()
    run_quant(case, variant, n_tok=shorter, codes=codes, scales=sc)
    short_out = run(case_slice(case, 0, shorter), variant)
    check_quant_bytes(short_out, codes, sc, shorter)
    assert torch.equal(codes[shorter:], before_codes[shorter:])
    assert torch.equal(reference.aiter_unshuffle_scales(sc)[shorter:], before_sc[shorter:])


@pytest.mark.parametrize("q_lens", [[1030, 517, 263], [129, 67, 35]], ids=["2r", "1r"])
@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_pack_epilogue_bytes(variant, hkv, q_lens):
    """``pack_out`` (the sequence-parallel all-to-all rows) is the reference pack of
    the kernel's bf16 output byte for byte, rows past T untouched."""
    require_gfx950()
    case = make_case(q_lens, [4000, 2100, 1300], hkv, (0.5, 2.0), seed=0)
    n_tok = case.q.shape[0]
    out = run(case, variant)
    pack = run_pack(case, variant)
    assert torch.equal(pack[:n_tok], reference.sp_pack_rows(out.view(n_tok, -1)))
    assert bool((pack[n_tok:] == FILL).all())


@pytest.mark.parametrize("hkv,n_tok", [(TP4_HKV, 65536)], ids=["tp4-64k"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_largest_step(variant, hkv, n_tok):
    """The largest served step (max-num-batched-tokens 65536 at TP4): the reference on
    sampled rows and the quant epilogue's bytes."""
    require_gfx950()
    case = make_case([n_tok], [n_tok + 4096], hkv, seed=0, picks="shared")
    out = run(case, variant)
    check_reference(case, out, variant, rows=case.rows(64))
    codes, sc = run_quant(case, variant)
    check_quant_bytes(out, codes, sc, n_tok)


# --- graph replay --------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["plain", "quant", "pack"])
@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_graph_replay(variant, hkv, mode):
    """One call captured in a HIP graph, replayed on three new inputs (Q, top-k,
    positions copied into the captured buffers) with the outputs refilled with NaN /
    0xA5 before each replay: every replay equals an eager call bit for bit, and the
    bf16 output matches the reference."""
    require_gfx950()
    q_lens, ctx = [1027, 514, 3, 262], [3000, 2100, 1100, 900]
    static = make_case(q_lens, ctx, hkv, seed=40)
    n_tok, h = static.q.shape[:2]
    padded = (n_tok + 31) // 32 * 32
    out = torch.empty_like(static.q)
    codes = torch.empty((padded, h * HEAD_DIM), dtype=torch.uint8, device="cuda")
    sc = torch.empty((padded, h * 4), dtype=torch.uint8, device="cuda")
    pack = torch.empty((n_tok, h * 132), dtype=torch.uint8, device="cuda")
    kw = {"plain": {}, "quant": {"quant_out": (codes, sc)}, "pack": {"pack_out": pack}}[mode]
    run(static, variant, out=out, **kw)  # compiles outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(static, variant, out=out, **kw)
    for seed in (41, 42, 43):
        new = make_case(q_lens, ctx, hkv, seed=seed)  # new queries and picks
        static.q.copy_(new.q)
        static.topk.copy_(new.topk)
        out.fill_(float("nan"))
        codes.fill_(FILL), sc.fill_(FILL), pack.fill_(FILL)
        graph.replay()
        torch.cuda.synchronize()
        eager = run(static, variant)  # the captured buffers, now holding the new inputs
        if mode == "plain":
            assert torch.equal(out, eager)
            check_reference(static, out, variant)
        elif mode == "quant":
            check_quant_bytes(eager, codes, sc, n_tok)
        else:
            assert torch.equal(pack, reference.sp_pack_rows(eager.view(n_tok, -1)))


# --- determinism next to other kernels -----------------------------------------------


@pytest.mark.parametrize("hkv,n_tok", [(TP4_HKV, 1024), (TP4_HKV, 4096)])
@pytest.mark.parametrize("variant", VARIANTS)
def test_bitwise_next_to_mfma_waves(variant, hkv, n_tok):
    """With another kernel's MFMA waves on every CU the output equals an undisturbed
    run's (a v_pk_mul_f32 in the dequant gave wrong products there, MI355X); one- and
    two-round layouts."""
    require_gfx950()
    from tests.mfma_spinner import mfma_spinner

    case = make_case([n_tok], [n_tok + 30000], hkv, seed=7)
    main, side = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)

    def on(stream):
        with torch.cuda.stream(stream):
            return run(case, variant)

    ref = on(main)
    torch.cuda.synchronize()
    spin = mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000, side)  # 2 ms, several times the attention
        out = on(main)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)


def test_supported_shapes():
    require_gfx950()
    supports = host.supports_nvfp4_sparse_prefill
    assert supports(16, 1, HEAD_DIM, TOPK, BLOCK)  # TP4
    assert supports(32, 2, HEAD_DIM, TOPK, BLOCK)
    assert not supports(8, 1, HEAD_DIM, TOPK, BLOCK)
    assert not supports(16, 1, HEAD_DIM, 8, BLOCK)
    assert not supports(16, 1, 64, TOPK, BLOCK)
    assert not supports(16, 1, HEAD_DIM, TOPK, 64)


# --- fp8 (experimental, not served) ---------------------------------------------------


@pytest.mark.parametrize("hkv", HKVS)
def test_fp8_matches_reference_with_fp8_operands(hkv):
    """The fp8 kernel against the fp64 attention of e4m3-rounded Q, K and V."""
    require_gfx950()
    case = make_case([129], [512], hkv, seed=29)
    out = run(case, "fp8")
    check_reference(case, out, "fp8", rows=[0, 17, 64, 128], fp8=True)


@pytest.mark.parametrize("n_tok", [1, 3, 257, 1025, 1537, 2048, 4097])
@pytest.mark.parametrize("hkv", HKVS)
def test_fp8_prepacked_query_bitwise(hkv, n_tok):
    """A bf16 Q (packed by ``query.pack_query`` in the dispatch) and the same Q
    prepacked by the caller give the same bits; the result is within the fp8
    contract."""
    require_gfx950()
    case = make_case([n_tok], [max(4096, n_tok)], hkv, seed=17)
    out = run(case, "fp8")
    q8 = pack_query(case.q)
    again = torch.full_like(out, float("nan"))
    run(case, "fp8", q=q8, out=again)
    assert torch.equal(again, out)
    check_reference(case, out, "fp8", rows=sorted({0, n_tok // 2, n_tok - 1}), fp8=True)


def test_fp8_query_packing_all_bf16_values():
    """``pack_query`` over all 65536 bf16 encodings: e4m3 of the value clamped to
    +-448 (NaN keeps its sign), bytes of each eight reordered to [0, 2, 4, 6, 1, 3, 5,
    7] -- checked element by element against an index formula."""
    require_gfx950()
    q = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.uint16)
    q = q.view(torch.bfloat16).reshape(-1, 16, 128)
    got = pack_query(q).reshape(-1)
    dst = torch.arange(q.numel(), device="cuda")
    src = dst // 8 * 8 + (dst % 4) * 2 + (dst % 8) // 4
    x = q.reshape(-1)[src].float()
    expected = x.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    nan = torch.isnan(x)
    sign = (x.view(torch.int32) >> 24) & 0x80
    expected = torch.where(nan, (sign | 0x7F).to(torch.uint8), expected)
    assert torch.equal(got, expected)


def test_fp8_lut_requires_eager_warmup(monkeypatch):
    """A cache miss during capture must not retain an uninitialized LUT."""
    require_gfx950()
    host._fp8_lut.cache_clear()
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Warm up FP8 prefill"):
        host._fp8_lut(device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    table = host._fp8_lut(device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert host._fp8_lut(device) is table


@pytest.mark.parametrize("hkv", HKVS)
@pytest.mark.parametrize("n_tok", [1, 33, 1024, 1025, 1537, 4096, 4097])
def test_fp8_output_bytes_and_graph(hkv, n_tok):
    require_gfx950()
    case = make_case([n_tok], [n_tok + 3000], hkv, seed=37, picks="shared")
    out = run(case, "fp8")
    for _ in range(2):
        assert torch.equal(run(case, "fp8"), out)
    pack = run_pack(case, "fp8")
    assert torch.equal(pack[:n_tok], reference.sp_pack_rows(out.reshape(n_tok, -1)))
    assert (pack[n_tok:] == FILL).all()
    codes, scales = run_quant(case, "fp8")
    check_quant_bytes(out, codes, scales, n_tok)
    assert (codes[n_tok:] == FILL).all()
    graph = torch.cuda.CUDAGraph()
    target = torch.empty_like(out)
    with torch.cuda.graph(graph):
        run(case, "fp8", out=target)
    target.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(target, out)


@pytest.mark.parametrize("hkv", [TP4_HKV])
def test_fp8_chunk_reference(hkv):
    require_gfx950()
    case = make_case([131, 2, 122], [3011, 1024, 2222], hkv, seed=7, picks="shared")
    out = run(case, "fp8")
    rows = case.rows(16)
    ref = reference.fp8_chunk_prefill_reference(*case.args(), rows=rows)
    err = reference.rel_l2(out[rows], ref)
    cos = torch.nn.functional.cosine_similarity(out[rows].double(), ref.double(), dim=-1).min().item()
    print(f"fp8 vs the chunk reference hkv={hkv}: rel L2 {err:.4e}, min cos {cos:.8f}")
    assert err < 2.5e-3
    assert cos > 0.99999


@pytest.mark.parametrize("n_tok", [256, 1024, 4096])
def test_fp8_bitwise_next_to_mfma(n_tok):
    test_bitwise_next_to_mfma_waves("fp8", TP4_HKV, n_tok)


@pytest.mark.parametrize(
    "variant,n_tok",
    [
        ("fp8", 129),
        ("fp8", 1025),
        ("fp8", 1537),
        ("fp8", 4097),
        ("bf16", 1025),
        pytest.param(
            "bf16",
            1537,
            marks=pytest.mark.xfail(
                strict=True,
                reason="the bf16 four-query tile turns a NaN V scale past the sequence end into NaN through zero P",
            ),
        ),
    ],
)
def test_nan_scales_past_length(variant, n_tok):
    """NaN scale bytes past the sequence end (unwritten tokens) leave the output finite and unchanged."""
    require_gfx950()
    case = make_case([n_tok], [n_tok + 17], TP4_HKV, seed=61, picks="shared")
    clean = run(case, variant)
    kc, ks, vc, vs = reference.unpack_kv_pages(case.cache, TP4_HKV)
    end = int(case.pos[-1]) + 1
    page = int(case.table[0, end // 128])
    tail = end % 128
    ks[page, :, tail:] = 0x7F
    vs[page, :, tail:] = 0x7F
    poisoned = reference.pack_kv_pages(kc, ks, vc, vs)
    dirty = run(dataclasses.replace(case, cache=poisoned), variant)
    assert torch.isfinite(dirty).all()
    assert torch.equal(clean, dirty)


@pytest.mark.parametrize("n_tok", [256, 1024])
def test_fp8_future_pick_before_table_load(n_tok):
    require_gfx950()
    case = make_case([n_tok], [n_tok + 3000], TP4_HKV, seed=83, picks="shared")
    topk = case.topk.clone()
    topk[:, :, -1] = -1
    clean = run(dataclasses.replace(case, topk=topk), "fp8")
    topk[:, :, -1] = 2**30
    future = run(dataclasses.replace(case, topk=topk), "fp8")
    assert torch.equal(clean, future)
