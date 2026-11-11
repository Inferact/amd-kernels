# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``nvfp4_sparse_decode(..., index_score=, index_max_seq_len=)``: the attention takes the indexer's scores
(``nvfp4_index_score_rows``) and writes ``topk_idx`` itself. Against the two-launch chain (``nvfp4_index_score_top16``
then the attention on its ids): the ids bit for bit and the attention output bytes equal, on the fused routes (one KV
head, top-16, bf16 or fp16, <= 256 rows: the rowxcd / prefetch / fixed kernels at ql 8 (fp16: rowxcd, else the
generic kernel), the generic kernel at ql 4; the selector body runs in every attention workgroup) and on the route that launches the selector first (> 256 rows).
Short rows (< 16 blocks: the exact fallback, ids from memory), rows past 1792 blocks (the window loop), padding
requests, 1-token rows, init / local blocks, a NaN index-q element, equal keys (ties), graph replay at other lengths,
determinism."""

import pytest
import torch
from amd_kernels.attention.nvfp4_sparse_decode_gqa16_d128_p128_top16 import host as ah
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16 import reference as iref
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import (
    nvfp4_index_score_rows,
    nvfp4_index_score_top16,
)

from .test_sparse_decode import require_gfx950

MAX = 1 << 20
BLK, D, H = 128, 128, 16


def build(lengths, ql, seed, nan_q=False, equal_keys=False):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    reqs = len(lengths)
    rows = reqs * ql
    nb = [max(1, -(-s // BLK)) for s in lengths]
    pages = 1 + sum(nb)
    keys = (torch.randn(pages * BLK, D, device="cuda", generator=g) * 2).bfloat16()
    if equal_keys:
        keys = keys[:1].expand(pages * BLK, D).contiguous()
    icache = iref.nvfp4_index_pages(keys)
    kv = torch.randint(0, 256, (pages, 2, BLK, 72), dtype=torch.uint8, device="cuda", generator=g) & 0x77
    bt = torch.zeros(reqs, -(-MAX // BLK), dtype=torch.int32, device="cuda")
    p = 1
    for r, n in enumerate(nb):
        bt[r, :n] = torch.randperm(n, device="cuda", generator=g).to(torch.int32) + p
        p += n
    seq = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    iq = (torch.randn(rows, 1, D, device="cuda", generator=g) * 0.5).to(torch.float8_e4m3fn)
    if nan_q:
        iq.view(torch.uint8)[rows // 2, 0, 7] = 0x7F
    q = torch.randn(rows, H, D, device="cuda", generator=g).bfloat16()
    return dict(iq=iq, icache=icache, kv=kv, bt=bt, seq=seq, q=q, ql=ql, rows=rows)


def chain(c, init, local, fused, ids, out, score, compute=torch.bfloat16):
    one = torch.ones((), dtype=torch.float32, device="cuda")
    sm = D ** -0.5
    if fused:
        nvfp4_index_score_rows(c["iq"], c["icache"], score, c["bt"], c["seq"], init, local,
                               query_len=c["ql"], max_seq_len=MAX)
        ah.nvfp4_sparse_decode(c["q"], c["kv"], ids, c["bt"], c["seq"], sm, out, k_scale=one,
                               v_scale=one, decode_query_len=c["ql"], index_score=score,
                               index_max_seq_len=MAX, compute_type=compute)
    else:
        nvfp4_index_score_top16(c["iq"], c["icache"], c["bt"], c["seq"], init, local, ids,
                                query_len=c["ql"], max_seq_len=MAX, score=score)
        ah.nvfp4_sparse_decode(c["q"], c["kv"], ids, c["bt"], c["seq"], sm, out, k_scale=one,
                               v_scale=one, decode_query_len=c["ql"], compute_type=compute)


def buffers(c):
    rows = c["rows"]
    return (torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda"),
            torch.full((rows, H, D), 7.0, dtype=torch.bfloat16, device="cuda"),
            torch.full((1, rows, iref.score_width(MAX)), float("nan"), dtype=torch.float32, device="cuda"))


def same(c, init, local, monkeypatch=None, compute=torch.bfloat16):
    ia, oa, sa = buffers(c)
    ib, ob, sb = buffers(c)
    chain(c, init, local, False, ia, oa, sa, compute)
    names = []
    if monkeypatch is not None:
        real = ah._run_compiled

        def spy(exe, *args):
            names.append(getattr(exe, "kernel_name", ""))
            return real(exe, *args)

        monkeypatch.setattr(ah, "_run_compiled", spy)
    chain(c, init, local, True, ib, ob, sb, compute)
    if monkeypatch is not None:
        monkeypatch.undo()
        fused = c["rows"] <= 256  # rowxcd / prefetch / fixed (ql 8) or generic (ql 4)
        assert any(n.endswith("_select16") for n in names) == fused, names
    assert torch.equal(ia, ib), (ia - ib).abs().max()
    assert torch.equal(oa.view(torch.int16), ob.view(torch.int16))
    return ib, ob


CASES = [
    (8, [32768]),
    (8, [131072 + 37]),
    (8, [200000, 5000]),
    (8, [0, 300, 1]),
    (8, [1, 129, 70000, 2047]),
    (8, [240000, 0, 1000, 196608]),  # 1875 blocks: the window loop
    (8, [1500, 16 * 128 + 9, 15 * 128 + 8, 8]),  # around 16 blocks: fallback / fast
    (4, [65536, 9]),  # ql 4: the generic kernel
    (4, [196608]),
    (4, [0, 3, 1900 * 128 + 3, 16 * 128]),
    (8, [40000] * 5),  # 40 rows: the prefetch kernel
    (8, [3000 + 997 * i for i in range(17)]),  # 136 rows: the fixed kernel
    (8, [2000 + 501 * i for i in range(31)] + [0]),  # 256 rows: prefetch, one split
    (8, [1500 + 37 * i for i in range(33)]),  # 264 rows: the selector launch
    (4, [9000 + 1001 * i for i in range(16)]),  # 64 rows at ql 4: generic
]


@pytest.mark.parametrize("compute", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("ql,lengths", CASES)
@pytest.mark.parametrize("init,local", [(0, 1), (2, 3)])
def test_ids_and_output_match_the_two_launch_chain(ql, lengths, init, local, compute, monkeypatch):
    require_gfx950()
    c = build(lengths, ql, seed=len(lengths) * 131 + lengths[0] % 977 + init)
    ids, _ = same(c, init, local, monkeypatch, compute)
    assert int((ids == -7).sum()) == 0


@pytest.mark.parametrize("kind", ["nan_q", "equal_keys"])
def test_exceptional_scores(kind):
    require_gfx950()
    c = build([70000, 3000, 1], 8, seed=5, nan_q=kind == "nan_q", equal_keys=kind == "equal_keys")
    same(c, 0, 1)


def test_deterministic():
    require_gfx950()
    c = build([150000, 20000, 0], 8, seed=9)
    ia, oa, sa = buffers(c)
    ib, ob, sb = buffers(c)
    chain(c, 0, 1, True, ia, oa, sa)
    chain(c, 0, 1, True, ib, ob, sb)
    torch.cuda.synchronize()
    assert torch.equal(ia, ib) and torch.equal(oa.view(torch.int16), ob.view(torch.int16))


@pytest.mark.parametrize("ql,reqs", [(8, 1), (8, 3), (8, 4), (4, 2)])
def test_graph_replay_at_other_lengths(ql, reqs):
    require_gfx950()
    first = [190000, 7000, 129, 64000][:reqs]
    later = [[0] * reqs, [1] * reqs, [100000, 0, 240000, 33][:reqs], [17 * 128] * reqs]
    c = build([max(a, max(L[i] for L in later)) for i, a in enumerate(first)], ql, seed=11 + reqs)
    c["seq"].copy_(torch.tensor(first, dtype=torch.int32))
    ib, ob, sb = buffers(c)
    chain(c, 0, 1, True, ib, ob, sb)  # warm (compile) before capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        chain(c, 0, 1, True, ib, ob, sb)
    for lens in [first] + later:
        c["seq"].copy_(torch.tensor(lens, dtype=torch.int32))
        ib.fill_(-7)
        ob.fill_(7.0)
        g.replay()
        ia, oa, sa = buffers(c)
        chain(c, 0, 1, False, ia, oa, sa)
        torch.cuda.synchronize()
        assert torch.equal(ia, ib), lens
        assert torch.equal(oa.view(torch.int16), ob.view(torch.int16)), lens


@pytest.mark.parametrize("quant", [False, True], ids=["bf16", "mxfp8"])
@torch.inference_mode()
def test_no_compile_after_first_call(quant, monkeypatch):
    """The fused kernel is one compile per output mode (rows are a runtime argument): after
    the first call (1 request), 1-4 requests at other lengths compile nothing; MXFP8 bytes
    equal the two-launch chain's."""
    require_gfx950()
    import flydsl.compiler as flyc

    from .test_sparse_decode import quant_buffers

    ah._get_kernel_select.cache_clear()
    c = build([150000, 3000, 70000, 0], 8, seed=21)
    one = torch.ones((), dtype=torch.float32, device="cuda")

    def call(reqs, select, ids, score, qb):
        rows = reqs * 8
        sub = dict(c, iq=c["iq"][:rows], q=c["q"][:rows], bt=c["bt"][:reqs], seq=c["seq"][:reqs], rows=rows)
        out = torch.empty(rows, H, D, dtype=torch.bfloat16, device="cuda")
        if select:
            nvfp4_index_score_rows(sub["iq"], sub["icache"], score, sub["bt"], sub["seq"], 0, 1,
                                   query_len=8, max_seq_len=MAX)
            ah.nvfp4_sparse_decode(sub["q"], sub["kv"], ids[:, :rows], sub["bt"], sub["seq"], D ** -0.5,
                                   out, k_scale=one, v_scale=one, decode_query_len=8,
                                   quant_out=(qb[0], qb[1], 16) if quant else None,
                                   index_score=score, index_max_seq_len=MAX)
        else:
            nvfp4_index_score_top16(sub["iq"], sub["icache"], sub["bt"], sub["seq"], 0, 1, ids[:, :rows],
                                    query_len=8, max_seq_len=MAX, score=score)
            ah.nvfp4_sparse_decode(sub["q"], sub["kv"], ids[:, :rows], sub["bt"], sub["seq"], D ** -0.5,
                                   out, k_scale=one, v_scale=one, decode_query_len=8,
                                   quant_out=(qb[0], qb[1], 16) if quant else None)
        return out

    rows = 32
    width = H * D
    score = torch.empty(1, rows, iref.score_width(MAX), dtype=torch.float32, device="cuda")
    ids_b = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    qb_b = quant_buffers(rows, width)
    call(1, True, ids_b, score, qb_b)  # first call: compiles
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile

    def counting(fn, *a, **k):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *a, **k)

    monkeypatch.setattr(flyc, "compile", counting)
    for reqs in (2, 3, 4, 1):
        c["seq"][:reqs].copy_(torch.tensor([150000, 3000, 70000, 0][:reqs], dtype=torch.int32) - 7 * reqs)
        c["seq"].clamp_(min=0)
        qb_b = quant_buffers(rows, width)
        ob = call(reqs, True, ids_b, score, qb_b)
        torch.cuda.synchronize()
        assert not compiled, (reqs, compiled)
        ids_a = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
        qb_a = quant_buffers(rows, width)
        oa = call(reqs, False, ids_a, torch.empty_like(score), qb_a)
        torch.cuda.synchronize()
        r = reqs * 8
        assert torch.equal(ids_a[:, :r], ids_b[:, :r]), reqs
        if quant:
            n = -(-r // 16) * 16
            assert torch.equal(qb_a[0][:n], qb_b[0][:n]), reqs
            assert torch.equal(qb_a[1], qb_b[1]), reqs
        else:
            assert torch.equal(oa.view(torch.int16), ob.view(torch.int16)), reqs
        compiled.clear()


@pytest.mark.parametrize("ql", [8, 4])
def test_wider_workspace_and_more_rows(ql):
    """A score workspace wider than score_width(max_seq_len) and with more rows than the batch (a
    caller's preallocated buffer), and a top-k buffer with spare rows: the same ids and output; the
    spare rows of topk_idx stay untouched."""
    require_gfx950()
    c = build([100000, 1700, 0, 64000][: 32 // ql // 2], ql, seed=31)
    rows = c["rows"]
    ia, oa, sa = buffers(c)
    chain(c, 0, 1, False, ia, oa, sa)
    wide = torch.full((1, rows + 16, iref.score_width(MAX) + 96), float("nan"), dtype=torch.float32, device="cuda")
    ib = torch.full((1, rows + 8, 16), -7, dtype=torch.int32, device="cuda")
    ob = torch.full((rows, H, D), 7.0, dtype=torch.bfloat16, device="cuda")
    chain(c, 0, 1, True, ib, ob, wide)
    torch.cuda.synchronize()
    assert torch.equal(ia, ib[:, :rows])
    assert bool((ib[:, rows:] == -7).all())
    assert torch.equal(oa.view(torch.int16), ob.view(torch.int16))


@pytest.mark.parametrize("compute", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("ql,max_reqs", [(8, 32), (4, 16)])
@torch.inference_mode()
def test_every_request_count_warms_once(ql, max_reqs, compute, monkeypatch):
    """vLLM warms the attention by calling every request count once (``_warm_attention``); with
    ``index_score`` passed there too, a second pass over every count compiles nothing (the fused
    kernels' keys: rowxcd, prefetch s4 / s2 / s1, fixed, generic)."""
    require_gfx950()
    import flydsl.compiler as flyc

    for get in (ah._get_kernel_select, ah._get_kernel_select_prefetch, ah._get_kernel_select_fixed,
                ah._get_kernel_select_generic):
        get.cache_clear()
    c = build([3000 + 77 * i for i in range(max_reqs)], ql, seed=41)
    rows = c["rows"]
    score = torch.empty(1, rows, iref.score_width(MAX), dtype=torch.float32, device="cuda")
    ids = torch.full((1, rows, 16), -7, dtype=torch.int32, device="cuda")
    out = torch.empty(rows, H, D, dtype=torch.bfloat16, device="cuda")
    one = torch.ones((), dtype=torch.float32, device="cuda")

    def call(reqs):
        r = reqs * ql
        nvfp4_index_score_rows(c["iq"][:r], c["icache"], score, c["bt"][:reqs], c["seq"][:reqs], 0, 1,
                               query_len=ql, max_seq_len=MAX)
        ah.nvfp4_sparse_decode(c["q"][:r], c["kv"], ids[:, :r], c["bt"][:reqs], c["seq"][:reqs], D ** -0.5,
                               out[:r], k_scale=one, v_scale=one, decode_query_len=ql,
                               index_score=score, index_max_seq_len=MAX, compute_type=compute)

    for reqs in range(1, max_reqs + 1):
        call(reqs)
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile

    def counting(fn, *a, **k):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *a, **k)

    monkeypatch.setattr(flyc, "compile", counting)
    for reqs in range(max_reqs, 0, -1):
        call(reqs)
    torch.cuda.synchronize()
    assert not compiled, compiled
