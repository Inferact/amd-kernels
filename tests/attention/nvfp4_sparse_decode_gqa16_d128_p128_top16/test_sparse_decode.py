# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The NVFP4 sparse decode against its contract (``reference.py``).

``test_contract`` runs the serving call (automatic schedule, the XCD count from the device) at one request count per
compiled variant of each served configuration (``plan.py``: ``SERVED``, ``variants``) and at the largest, for the bf16
and the MXFP8 output: against the fp64 reference, three launches bit for bit, graph padding requests zero, rows past
the batch untouched. The other tests cover explicit schedules (every split count, with and without the XCD grid,
caches of random bytes), CUDA graph replay, bytes past the lengths and strided pages.
"""

import pytest
import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import nvfp4_sparse_decode
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.device import initialize_xcd_topology
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16.reference import (
    BLOCK,
    COS,
    GROUP,
    HEAD_DIM,
    QUANT_REL,
    REL_L2,
    TOPK,
    aiter_scale_offsets,
    cosine,
    layout_offsets,
    mxfp8_dequantize,
    mxfp8_quantize,
    pack_kv,
    quantize_cache,
    rel_l2,
    sparse_decode_reference,
    tile_code_offsets,
    unpack_kv,
)

from .plan import CUS, SERVED, XCDS, plan, variants

SM_SCALE = HEAD_DIM**-0.5
K_SCALE, V_SCALE = 0.5, 2.0
FILL = 0xA5  # quant buffers' bytes the kernel must not write


def require_gfx950():
    if (
        not torch.cuda.is_available()
        or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    ):
        pytest.skip("requires gfx950")
    if torch.cuda.get_device_properties(0).multi_processor_count != CUS:
        pytest.skip(f"the plan tables assume {CUS} CUs (MI355X)")


def lengths_for(reqs, seed, pad=0):
    """Context lengths: short and long ones, then reproducible ones in [129, 120000]; the last ``pad`` requests are
    CUDA graph padding (length 0)."""
    gen = torch.Generator().manual_seed(seed)
    base = [9, 300, 2100, 20000, 131, 4096, 777, 65]
    extra = torch.randint(129, 120001, (max(0, reqs - len(base)),), generator=gen)
    lengths = (base + extra.tolist())[:reqs]
    return lengths[: reqs - pad] + [0] * pad


def selections(lengths, ql, hkv, gen):
    """``[hkv, rows, TOPK]`` logical blocks: the row's own block, ten older blocks the request's rows share, three of
    the row's own, one block past the request's end; -1 fills the rest. KV head h rolls the list by h."""
    rows = len(lengths) * ql
    topk = torch.full((1, rows, TOPK), -1, dtype=torch.int32)
    for r, s in enumerate(lengths):
        s = max(s, 1)
        nb = (s + BLOCK - 1) // BLOCK
        base = torch.randperm(nb, generator=gen).to(torch.int32)
        for q in range(ql):
            cur = max(s - ql + q, 0) // BLOCK
            older = base[(base != cur) & (base <= cur)]
            pick = torch.cat([older[:10], older[10 + 3 * q : 13 + 3 * q]])
            sel = torch.cat([torch.tensor([cur]), pick, torch.tensor([nb + 5])])
            sel = sel[:TOPK].to(torch.int32)
            topk[0, r * ql + q, : sel.numel()] = sel
    return torch.cat([topk.roll(h, -1) for h in range(hkv)])


def make_case(hkv, ql, lengths, seed, random_bytes=False):
    """A decode batch. Only the pages some row selects are materialized; the block table maps a request's selected
    blocks to distinct random pages (page 0 is never selected)."""
    gen = torch.Generator().manual_seed(seed)
    topk = selections(lengths, ql, hkv, gen)
    reqs = len(lengths)
    widths = [(max(s, 1) + BLOCK - 1) // BLOCK for s in lengths]
    used = []
    for r, w in enumerate(widths):
        sel = topk[:, r * ql : (r + 1) * ql]
        used.append(sorted({b for b in sel.flatten().tolist() if 0 <= b < w}))
    pages = 1 + sum(len(u) for u in used)
    perm = torch.randperm(pages - 1, generator=gen) + 1
    table = torch.zeros(reqs, max(widths), dtype=torch.int32)
    i = 0
    for r, blocks in enumerate(used):
        for b in blocks:
            table[r, b] = perm[i]
            i += 1
    if random_bytes:
        cache = torch.randint(0, 256, (pages, 2 * hkv, BLOCK, 72), generator=gen)
        cache = cache.to(torch.uint8)
        for side in range(2):
            start = side * hkv * 9216 + hkv * 8192
            scales = cache.view(pages, -1)[:, start : start + hkv * 1024]
            scales.masked_fill_((scales & 0x7F) == 0x7F, 0x38)  # no NaN scales
        cache = cache.cuda()
    else:
        chunks = []
        for p0 in range(0, pages, 256):
            shape = (min(256, pages - p0), BLOCK, hkv, HEAD_DIM)
            k = (torch.randn(shape, generator=gen) * 2).bfloat16().cuda()
            v = (torch.randn(shape, generator=gen) * 2).bfloat16().cuda()
            chunks.append(quantize_cache(k, v, K_SCALE, V_SCALE))
        cache = torch.cat(chunks)
    q = torch.randn(reqs * ql, hkv * GROUP, HEAD_DIM, generator=gen)
    q = (q * (0.01 if random_bytes else 1.0)).bfloat16().cuda()
    return {
        "q": q,
        "kv_cache": cache,
        "topk_idx": topk.cuda(),
        "block_table": table.cuda(),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32).cuda(),
        "ql": ql,
        # fp32 device scalars, made here: a launch under graph capture copies nothing
        "k_scale": torch.tensor(K_SCALE, dtype=torch.float32, device="cuda"),
        "v_scale": torch.tensor(V_SCALE, dtype=torch.float32, device="cuda"),
    }


def quant_buffers(rows, width):
    rows16, rows32 = -(-rows // 16) * 16, -(-rows // 32) * 32
    codes = torch.full((rows16 + 16, width), FILL, dtype=torch.uint8, device="cuda")
    sc = torch.full((rows32 + 32, width // 32), FILL, dtype=torch.uint8, device="cuda")
    return codes, sc


def run(case, compute, out=None, quant=None, **kw):
    """The serving call; returns the bf16 output (with ``quant`` = (codes, scales, tile) those buffers are written)."""
    q = case["q"]
    if out is None:
        out = torch.full_like(q, float("nan"))
    nvfp4_sparse_decode(
        q,
        case["kv_cache"],
        case["topk_idx"],
        case["block_table"],
        case["seq_lens"],
        SM_SCALE,
        out,
        k_scale=case["k_scale"],
        v_scale=case["v_scale"],
        decode_query_len=case["ql"],
        quant_out=quant,
        compute_type=compute,
        **kw,
    )
    return out


def reference(case, compute):
    return sparse_decode_reference(
        case["q"],
        case["kv_cache"],
        case["topk_idx"],
        case["block_table"],
        case["seq_lens"],
        SM_SCALE,
        K_SCALE,
        V_SCALE,
        case["ql"],
        round_kv_to_e4m3=compute == torch.bfloat16,
    )


def live_rows(case):
    return (case["seq_lens"] > 0).repeat_interleave(case["ql"])


def check_bf16(name, out, ref, live):
    assert torch.isfinite(out).all(), name
    rel, cos = rel_l2(out[live], ref[live]), cosine(out[live], ref[live])
    print(f"{name}: rel L2 {rel:.2e}, cos {cos:.7f}")
    assert rel < REL_L2 and cos > COS, (name, rel, cos)
    assert (out[~live] == 0).all(), name


def read_quant(codes, sc, rows, width, tile):
    """The real rows' codes ``[rows, width]`` and scales ``[rows, width // 32]``."""
    if tile:
        got = codes.view(-1)[tile_code_offsets(rows, width, codes.device)]
    else:
        got = codes[:rows]
    return got, sc.view(-1)[aiter_scale_offsets(rows, width, sc.device)]


def check_quant_untouched(name, codes, sc, rows, width, tile):
    rows16, rows32 = -(-rows // 16) * 16, -(-rows // 32) * 32
    if tile:
        pad = tile_code_offsets(rows16 + 16, width, codes.device)[rows:]
        assert (codes.view(-1)[pad] == FILL).all(), name
    else:
        assert (codes[rows:] == FILL).all(), name
    pad = aiter_scale_offsets(rows32 + 32, width, sc.device)[rows:]
    assert (sc.view(-1)[pad] == FILL).all(), name


def check_quant(name, quant, ref, live, rows, width):
    codes, sc, tile = quant
    got, got_s = read_quant(codes, sc, rows, width, tile)
    deq = mxfp8_dequantize(got, got_s).view_as(ref)
    assert torch.isfinite(deq).all(), name
    row_rel = (deq - ref).flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1)
    worst = float(row_rel[live].max())
    print(f"{name}: MXFP8 worst row rel L2 {worst:.2e}")
    assert worst < QUANT_REL, (name, worst)
    assert (deq[~live] == 0).all(), name
    check_quant_untouched(name, codes, sc, rows, width, tile)
    return got, got_s


def same_plan_kwargs(key):
    """Arguments that make a bf16-output launch take ``key``'s schedule."""
    splits, _, xcds, _, _, tail = key[:6]
    if tail:
        return {"tail_splits": splits, "xcd_count": xcds}
    return {"splits": splits, "xcd_count": xcds or XCDS}


CONTRACT = []
for _config in SERVED:
    for _quant in (False, True):
        _extra = {2, 3, 4} if _config in ("kv1_bf16_ql8", "kv1_f16_ql8") else set()
        _reqs = sorted(set(variants(_config, _quant).values()) | {SERVED[_config][3]} | _extra)
        CONTRACT += [(_config, _quant, r) for r in _reqs]


@pytest.mark.parametrize(
    "config,quant,reqs",
    CONTRACT,
    ids=[f"{c}-{'mxfp8' if q else 'bf16'}-r{r}" for c, q, r in CONTRACT],
)
@torch.inference_mode()
def test_contract(config, quant, reqs):
    """The serving call against the contract; one request count per compiled variant, padding requests (length 0)
    at the end of batches of three or more."""
    require_gfx950()
    hkv, compute, ql, _ = SERVED[config]
    case = make_case(hkv, ql, lengths_for(reqs, reqs, pad=int(reqs >= 3)), seed=reqs)
    rows, width = case["q"].shape[0], hkv * GROUP * HEAD_DIM
    key = plan(reqs, ql, hkv, compute, quant)
    name = f"{config} {'mxfp8' if quant else 'bf16'} R={reqs} key={key}"
    ref = reference(case, compute)
    live = live_rows(case)
    if not quant:
        big = torch.full((rows + 3, *case["q"].shape[1:]), float("nan"), device="cuda")
        big = big.bfloat16()
        outs = [run(case, compute, out=big[:rows].clone()) for _ in range(3)]
        out = run(case, compute, out=big[:rows])
        torch.cuda.synchronize()
        assert big[rows:].isnan().all(), name
        check_bf16(name, out, ref, live)
        for o in outs:
            assert torch.equal(o, out), name
        return
    tile = 16 if reqs % 2 else 0  # both code layouts of the o_proj input
    runs = []
    for _ in range(3):
        codes, sc = quant_buffers(rows, width)
        run(case, compute, out=torch.empty_like(case["q"]), quant=(codes, sc, tile))
        runs.append((codes, sc))
    torch.cuda.synchronize()
    got, got_s = check_quant(name, (*runs[0], tile), ref, live, rows, width)
    for codes, sc in runs[1:]:
        assert torch.equal(codes, runs[0][0]) and torch.equal(sc, runs[0][1]), name
    # the bytes are the MXFP8 quant of the same schedule's bf16 output
    out = run(case, compute, **same_plan_kwargs(key))
    torch.cuda.synchronize()
    want, want_s = mxfp8_quantize(out.view(rows, width))
    assert torch.equal(got, want) and torch.equal(got_s, want_s), name


EXPLICIT = [
    (compute, ql, reqs, splits, random_bytes, xcd)
    for compute in (torch.bfloat16, torch.float16)
    for ql, reqs in ((1, 6), (4, 6), (8, 6), (8, 4))
    for splits in (None, 1, 2, 4, 8)
    for random_bytes in (False, True)
    for xcd in (0, XCDS)
]


@pytest.mark.parametrize(
    "compute,ql,reqs,splits,random_bytes,xcd",
    EXPLICIT,
    ids=[f"{str(a)[6:]}-ql{b}-r{c}-s{d}-{'random' if e else 'quantized'}-xcd{f}" for a, b, c, d, e, f in EXPLICIT],
)
@torch.inference_mode()
def test_explicit_schedules_one_kv_head(compute, ql, reqs, splits, random_bytes, xcd):
    """1 KV head, bf16 and fp16 compute: every split count with and without the XCD grid (6 requests: a padded
    grid; 4 requests of 8 rows: the rowxcd kernels), with realistic caches and with caches of random bytes (every code
    and every finite scale byte)."""
    require_gfx950()
    lengths = [9, 300, 2100, 20000, 131, 4096][:reqs]
    case = make_case(1, ql, lengths, seed=0, random_bytes=random_bytes)
    outs = [run(case, compute, splits=splits, xcd_count=xcd) for _ in range(2)]
    ref = reference(case, compute)
    torch.cuda.synchronize()
    check_bf16(f"{compute} ql{ql} s{splits} xcd{xcd}", outs[0], ref, live_rows(case))
    assert torch.equal(outs[0], outs[1])


OTHER = [
    (hkv, compute, ql, splits, xcd)
    for hkv, compute in ((1, torch.float16),)
    for ql, splits, xcd in ((1, 1, 0), (4, 4, XCDS), (8, 4, 0))
]


@pytest.mark.parametrize(
    "hkv,compute,ql,splits,xcd",
    OTHER,
    ids=[f"kv{a}-{str(b)[6:]}-ql{c}-s{d}-xcd{e}" for a, b, c, d, e in OTHER],
)
@torch.inference_mode()
def test_fp16_explicit_splits_with_padding(hkv, compute, ql, splits, xcd):
    """fp16 on 1 KV head at explicit splits, padded and plain grids, with a graph padding request."""
    require_gfx950()
    lengths = [131, 4097, 2100, 777, 65, 0]
    case = make_case(hkv, ql, lengths, seed=42)
    out = run(case, compute, splits=splits, xcd_count=xcd)
    again = run(case, compute, splits=splits, xcd_count=xcd)
    ref = reference(case, compute)
    torch.cuda.synchronize()
    check_bf16(f"kv{hkv} {compute} ql{ql}", out, ref, live_rows(case))
    assert torch.equal(out, again)


QUANT_BYTES = [
    (hkv, ql, splits, xcd, tile, compute)
    for hkv, ql, splits, xcd, tile in (
        (1, 1, 1, 0, 0),
        (1, 4, 4, XCDS, 16),
    )
    for compute in (torch.bfloat16, torch.float16)
]


@pytest.mark.parametrize(
    "hkv,ql,splits,xcd,tile,compute",
    QUANT_BYTES,
    ids=[f"kv{a}-ql{b}-s{c}-xcd{d}-tile{e}-{str(f)[6:]}" for a, b, c, d, e, f in QUANT_BYTES],
)
@torch.inference_mode()
def test_quant_bytes(hkv, ql, splits, xcd, tile, compute):
    """``quant_out``: the codes (row-major or (16, 16) tiles) and AITER-layout scales are ``mxfp8_quantize`` of the
    bf16 output, byte for byte; rows past the batch are not written."""
    require_gfx950()
    lengths = [9, 300, 2100, 20000, 131, 4096, 777]
    case = make_case(hkv, ql, lengths, seed=0)
    rows, width = case["q"].shape[0], hkv * GROUP * HEAD_DIM
    kw = dict(splits=splits, xcd_count=xcd)
    out = run(case, compute, **kw)
    codes, sc = quant_buffers(rows, width)
    run(case, compute, out=torch.empty_like(out), quant=(codes, sc, tile), **kw)
    torch.cuda.synchronize()
    got, got_s = read_quant(codes, sc, rows, width, tile)
    want, want_s = mxfp8_quantize(out.view(rows, width))
    assert torch.equal(got, want) and torch.equal(got_s, want_s)
    check_quant_untouched("quant", codes, sc, rows, width, tile)


GRAPH = [(c, q) for c in SERVED for q in (False, True)]


@pytest.mark.parametrize(
    "config,quant", GRAPH, ids=[f"{c}-{'mxfp8' if q else 'bf16'}" for c, q in GRAPH]
)
@torch.inference_mode()
def test_graph_replay(config, quant):
    """The captured serving call (split tickets reset by the kernel) writes the eager bytes on every replay into
    outputs filled with NaN / 0xA5, and reads its inputs live: new q and cache values copied in place give the new
    eager result."""
    require_gfx950()
    hkv, compute, ql, max_reqs = SERVED[config]
    reqs = min(13, max_reqs)
    lengths = lengths_for(reqs, 7, pad=2)
    a, b = make_case(hkv, ql, lengths, seed=7), make_case(hkv, ql, lengths, seed=7)
    b["q"] = (torch.randn_like(a["q"].float()) * 1.5).bfloat16()
    b["kv_cache"] = torch.randint_like(a["kv_cache"], 0, 256)
    for side in range(2):
        start = side * hkv * 9216 + hkv * 8192
        sc_b = b["kv_cache"].view(b["kv_cache"].shape[0], -1)[:, start : start + hkv * 1024]
        sc_b.masked_fill_((sc_b & 0x7F) == 0x7F, 0x30)
    rows, width = reqs * ql, hkv * GROUP * HEAD_DIM

    def call(case, out, bufs):
        run(case, compute, out=out, quant=(*bufs, 16) if quant else None)

    def eager(case):
        out, bufs = torch.full_like(case["q"], float("nan")), quant_buffers(rows, width)
        call(case, out, bufs)
        return out, bufs

    want_a, want_b = eager(a), eager(b)
    g_out, g_bufs = torch.empty_like(a["q"]), quant_buffers(rows, width)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(a, g_out, g_bufs)
    for want, src in ((want_a, None), (want_a, None), (want_b, b)):
        if src is not None:
            a["q"].copy_(src["q"])
            a["kv_cache"].copy_(src["kv_cache"])
        g_out.fill_(float("nan"))
        for t in g_bufs:
            t.fill_(FILL)
        graph.replay()
        torch.cuda.synchronize()
        if quant:
            assert torch.equal(g_bufs[0], want[1][0]) and torch.equal(g_bufs[1], want[1][1])
        else:
            assert torch.equal(g_out, want[0])


def poison_tail(case, hkv):
    """Random codes and NaN block scales (0x7F) in every K / V slot of each request's last block past its length:
    the bytes an unwritten, reused slot may hold."""
    cache = case["kv_cache"]
    k_codes, k_scales, v_codes, v_scales = unpack_kv(cache)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for r, s in enumerate(case["seq_lens"].tolist()):
        t0 = s % BLOCK
        if s == 0 or t0 == 0:
            continue
        page = int(case["block_table"][r, (s - 1) // BLOCK])
        for codes in (k_codes, v_codes):
            codes[page, :, t0:] = torch.randint(
                0, 16, codes[page, :, t0:].shape, generator=gen, device="cuda"
            ).to(torch.uint8)
        k_scales[page, :, t0:] = 0x7F
        v_scales[page, :, t0:] = 0xFF
        v_scales[page, :, t0::2] = 0x7F
    cache.copy_(pack_kv(k_codes, k_scales, v_codes, v_scales))


POISON = [
    (hkv, compute, ql, reqs)
    for hkv, compute in ((1, torch.bfloat16), (1, torch.float16))
    for ql, reqs in ((1, 6), (4, 6), (8, 6), (8, 4))
]


@pytest.mark.parametrize(
    "hkv,compute,ql,reqs", POISON, ids=[f"kv{a}-{str(b)[6:]}-ql{c}-r{d}" for a, b, c, d in POISON]
)
@torch.inference_mode()
def test_bytes_past_the_length_are_ignored(hkv, compute, ql, reqs):
    """NaN scales and random codes past each request's length (its last block, a row's own block) leave the output
    finite and bit for bit the clean cache's (4 requests of 8 rows: the rowxcd kernels at 1 KV head)."""
    require_gfx950()
    lengths = [131, 4097, 2100, 777, 65, 300][:reqs]
    case = make_case(hkv, ql, lengths, seed=3)
    clean = run(case, compute, xcd_count=XCDS)
    poison_tail(case, hkv)
    dirty = run(case, compute, xcd_count=XCDS)
    torch.cuda.synchronize()
    assert torch.isfinite(dirty).all()
    assert torch.equal(clean, dirty)


@pytest.mark.parametrize("compute", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("reqs", [1, 2, 4, 8, 16, 20, 32, 64])
@torch.inference_mode()
def test_bitwise_next_to_mfma_waves(reqs, compute):
    """The TP4 paths (1 KV head, 8 rows per request) with a high-priority MFMA kernel (``tests/mfma_spinner.py``) on
    the same CUs: the output stays bit for bit the same (gfx950 has given wrong VALU products next to MFMA waves)."""
    from tests.mfma_spinner import mfma_spinner

    require_gfx950()
    case = make_case(1, 8, lengths_for(reqs, seed=31, pad=1 if reqs > 1 else 0), seed=31)
    clean = run(case, compute)
    main, side = torch.cuda.Stream(), torch.cuda.Stream(priority=-1)
    spin = mfma_spinner(torch.cuda.get_device_properties(0).multi_processor_count)
    for _ in range(8):
        spin(200_000, side)
        with torch.cuda.stream(main):
            got = run(case, compute)
        torch.cuda.synchronize()
        assert torch.equal(got, clean)


@pytest.mark.parametrize("hkv,ql,repeats,compute", [
    (1, 1, 1, torch.bfloat16), (1, 4, 1, torch.bfloat16),
    (1, 8, 2, torch.bfloat16), (1, 8, 5, torch.bfloat16),
    (1, 4, 1, torch.float16), (1, 8, 1, torch.float16), (1, 8, 5, torch.float16),
])
@torch.inference_mode()
def test_strided_pages(hkv, ql, repeats, compute):
    """Under BLHNC a layer's page is one slice of a wider block (stride(0) spans the neighbouring layers' pages, here
    NaN bytes): the decode reads the same bytes as from a dense cache."""
    require_gfx950()
    case = make_case(hkv, ql, [300, 2100, 20000, 131] * repeats, seed=2)
    dense = run(case, compute)
    cache = case["kv_cache"]
    page = cache[0].numel()
    wide = torch.full((cache.shape[0], 3 * page), 0x7F, dtype=torch.uint8, device="cuda")
    wide_cache = wide[:, page : 2 * page].view(cache.shape)
    wide_cache.copy_(cache)
    case["kv_cache"] = wide_cache
    strided = run(case, compute)
    torch.cuda.synchronize()
    assert torch.isfinite(dense).all() and torch.equal(dense, strided)


@pytest.mark.parametrize("ql", [1, 4, 8])
@pytest.mark.parametrize("splits", [1, 4])
@torch.inference_mode()
def test_padding_requests(ql, splits):
    """Graph padding requests (length 0) between real ones and an all-padding batch: the real rows match the
    reference, the padding rows are zero."""
    require_gfx950()
    lengths = [700, 0, 3000, 0, 0, 129]
    case = make_case(1, ql, lengths, seed=1)
    compute = torch.bfloat16
    out = run(case, compute, splits=splits)
    ref = reference(case, compute)
    torch.cuda.synchronize()
    check_bf16("padding", out, ref, live_rows(case))
    case["seq_lens"] = torch.zeros_like(case["seq_lens"])
    pad = run(case, compute, splits=splits)
    torch.cuda.synchronize()
    assert (pad == 0).all()


def test_layout_round_trip():
    """``unpack_kv`` inverts ``pack_kv`` on random codes and scales, and each layout map is a permutation of its
    region."""
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    for side in range(2):
        data, shift, scale = layout_offsets(side, "cuda")
        key = data * 2 + shift // 4
        assert torch.equal(key.flatten().sort().values, torch.arange(16384, device="cuda"))
        assert torch.equal(scale.flatten().sort().values, torch.arange(1024, device="cuda"))
    gen = torch.Generator(device="cuda").manual_seed(0)
    parts = [
        torch.randint(0, n, (3, 2, BLOCK, d), generator=gen, device="cuda").to(torch.uint8)
        for n, d in ((16, HEAD_DIM), (256, 8), (16, HEAD_DIM), (256, 8))
    ]
    for a, b in zip(unpack_kv(pack_kv(*parts)), parts):
        assert torch.equal(a, b)


def test_xcd_topology():
    require_gfx950()
    assert initialize_xcd_topology() == XCDS
