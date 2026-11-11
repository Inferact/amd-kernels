# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The index top-k (``topk``) against ``reference.py``: indices item by item, and the attend's page table and
context bounds, on realistic and adversarial score rows (ties, NaN, +-0 / denormals, sentinels, +-inf, rows past
max_seq_len, padding requests, multi-window rows), decode and ragged prefill rows (the page table: prefill rows),
single and split launches, persistent output buffers, graph replay and determinism. The NVFP4 indexer serves the
indices-only variants; the page-table variant serves the FP8 main-KV attend.
"""

import math

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    page_table_reference,
    score_width,
    topk_reference,
)
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import (
    host,
    index_topk_decode_indices,
    index_topk_prefill,
    index_topk_prefill_indices,
)

from .inputs import require_gfx950

KINDS = ["real", "ties", "equal", "neginf", "nan", "zeros", "sentinels", "inf", "rising"]


def make_scores(kind, heads, rows, width, nblk, gen):
    """Score rows of one kind; blocks past a row's count hold garbage (big values, NaN) the kernel must not read."""
    s = torch.randn(heads, rows, width, generator=gen)
    if kind == "real":  # a few outliers, the init block 1e30, the last two 1e29
        s = torch.where(torch.rand(s.shape, generator=gen) < 4.0 / width, s + 6, s)
        for r in range(rows):
            b = min(int(nblk[r]), width)
            if b > 0:
                s[:, r, max(b - 2, 0) : b] = 1e29
                s[:, r, 0] = 1e30
    elif kind == "ties":
        s = torch.randint(-3, 4, s.shape, generator=gen).float()
    elif kind == "equal":
        s = torch.full_like(s, 0.5)
    elif kind == "neginf":
        s = torch.full_like(s, float("-inf"))
    elif kind == "nan":
        s[torch.rand(s.shape, generator=gen) < 0.3] = float("nan")
        neg_nan = torch.tensor([-4194303], dtype=torch.int32).view(torch.float32)
        s[torch.rand(s.shape, generator=gen) < 0.1] = neg_nan.item()
    elif kind == "zeros":  # +0 / -0 / denormals
        vals = torch.tensor([0.0, -0.0, 1e-40, -1e-40, 1e-45, -1e-45])
        s = vals[torch.randint(0, 6, s.shape, generator=gen)]
    elif kind == "sentinels":
        s = torch.where(torch.rand(s.shape, generator=gen) < 0.2, 1e29, s)
        s = torch.where(torch.rand(s.shape, generator=gen) < 0.05, 1e30, s)
    elif kind == "inf":
        r = torch.rand(s.shape, generator=gen)
        s = torch.where(r < 0.1, float("inf"), s)
        s = torch.where((r > 0.1) & (r < 0.2), float("-inf"), s)
    elif kind == "rising":  # increasing with position
        s = torch.arange(width, dtype=torch.float32).expand(heads, rows, width).clone()
    tail = torch.randn(s.shape, generator=gen) * 1e6
    tail[torch.rand(s.shape, generator=gen) < 0.1] = float("nan")
    live = torch.arange(width)[None, None, :] < nblk[None, :, None]
    return torch.where(live, s, tail)


def outputs(heads, rows, nkv, pad):
    """topk_idx / sparse_bt / sparse_ctx filled with -7, and the tensors compared. ``pad``: views into larger
    persistent buffers (non-zero row starts, a larger head stride for topk_idx); the whole buffers are compared."""
    full = lambda shape: torch.full(shape, -7, dtype=torch.int32, device="cuda")  # noqa: E731
    if not pad:
        topk, sbt, sctx = full((heads, rows, 16)), full((rows * nkv, 128)), full((rows * nkv,))
        return (topk, sbt, sctx), (topk, sbt, sctx), (slice(None), slice(None))
    tb, bb, cb = (
        full((heads, rows + 37, 16)),
        full((rows * nkv + 64, 128)),
        full((rows * nkv + 64,)),
    )
    views = (tb[:, 5 : 5 + rows], bb[16 : 16 + rows * nkv], cb[16 : 16 + rows * nkv])
    return views, (tb, bb, cb), (slice(5, 5 + rows), slice(16, 16 + rows * nkv))


def expected(compared, where, ref_topk, ref_table):
    """The compared buffers as they must be after the call: -7 outside the views."""
    tb, bb, cb = (torch.full_like(t, -7) for t in compared)
    tb[:, where[0]] = ref_topk
    if ref_table is not None:
        bb[where[1]] = ref_table[0]
        cb[where[1]] = ref_table[1]
    return tb, bb, cb


def split_limit(score, nkv, rows_per_run, unsliced_bytes=0):
    """A span limit that splits a case into several runs per head: a third of the score's span, but at least what
    one run of ``rows_per_run`` rows and an unsliced block table need."""
    per_row = 4 * max(score.stride(1), 16, nkv * 128)
    floor = max(2 * per_row * rows_per_run + 1, unsliced_bytes + 1, 1 << 12)
    return max(host._span_bytes(score) // 3, floor)


SHAPES = [
    # reqs, query_len, max_seq_len, (lo, hi) seq_len, num_kv_heads, index heads
    (40, 4, 1 << 20, (30000, 210000), 1, 1),  # a long-context decode batch, 1 head
    (8, 4, 262144, (1, 262144), 1, 1),
    (48, 6, 262144, (1, 262144), 1, 1),  # six rows
    (5, 4, 1 << 20, (1 << 19, 1 << 20), 1, 1),  # 4 windows of 2048 blocks
    (16, 1, 1 << 20, (2040 * BLK, 2100 * BLK), 1, 1),  # the window edge
    (12, 1, 4000, (1, 4000), 1, 1),
    (6, 4, 2000, (0, 2000), 2, 1),
    (4, 2, 300000, (1, 300000), 2, 2),
    (7, 4, 20000, (20000, 40000), 1, 1),  # rows past max_seq_len
    (256, 4, 1 << 20, (1, 3000), 1, 1),  # 1024 short rows
    (24, 4, 200000, (1, 200000), 2, 1),  # one index head shared by 2 KV heads
    (32, 6, 1 << 20, (100000, 1 << 20), 1, 1),
    (16, 6, 200000, (1, 200000), 1, 1),
    (32, 6, 204800, (190000, 204800), 1, 1),
    (48, 8, 262144, (1, 262144), 1, 1),  # eight rows
    (48, 8, 1 << 20, (100000, 1 << 20), 1, 1),  # an eight-row decode batch at 1M
    (16, 8, 200000, (1, 200000), 1, 1),
    # rows past max_seq_len inside the padded width (2048 blocks): the window sized by max_seq_len goes on
    (8, 1, 200000, (200000, 262000), 1, 1),
    (4, 2, 0, (0, 0), 1, 1),  # a zero max_seq_len (dummy batch)
    (1, 1, 1, (0, 1), 1, 1),
]
SHAPE_IDS = [f"n{s[0]}q{s[1]}-msl{s[2]}-kv{s[4]}h{s[5]}-{i}" for i, s in enumerate(SHAPES)]


def decode_case(kind, shape):
    reqs, ql, msl, (lo, hi), nkv, heads = shape
    gen = torch.Generator().manual_seed(100 * KINDS.index(kind) + SHAPES.index(shape))
    seq = torch.randint(lo, hi + 1, (reqs,), generator=gen)
    if reqs > 1:
        seq[-1] = 0  # a cudagraph padding request
    tok = torch.arange(reqs * ql) % ql
    kv = (seq.repeat_interleave(ql) - ql + tok + 1).clamp(min=0)
    width = score_width(msl)
    score = make_scores(kind, heads, reqs * ql, width, (kv + BLK - 1) // BLK, gen)
    # the table covers every row's blocks (rows may run past max_seq_len)
    cols = max(1, math.ceil(max(msl, int(seq.max())) / BLK))
    bt = torch.randint(0, 1 << 22, (reqs, cols), generator=gen)
    req = torch.arange(reqs * ql) // ql
    return dict(
        score=score.cuda(),
        bt=bt.to(torch.int32).cuda(),
        seq=seq.to(torch.int32).cuda(),
        msl=msl,
        ql=ql,
        nkv=nkv,
        kv=kv.cuda(),
        req=req.cuda(),
        nblk=((kv + BLK - 1) // BLK).cuda(),
    )


def decode_reference(c):
    return topk_reference(c["score"], c["nblk"], c["msl"])


def check_all(compared, want):
    for name, g, w in zip(("topk_idx", "sparse_bt", "sparse_ctx"), compared, want):
        torch.testing.assert_close(g, w, rtol=0, atol=0, msg=name)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("rank_owned_order", [False, True])
@torch.inference_mode()
def test_topk_decode_indices(kind, shape, rank_owned_order):
    """Decode rows, indices only (the NVFP4 indexer's decode top-k)."""
    require_gfx950()
    c = decode_case(kind, shape)
    heads, rows, _ = c["score"].shape
    topk = torch.full((heads, rows, 16), -7, dtype=torch.int32, device="cuda")
    score = c["score"]
    if rank_owned_order:
        from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import rank_owned_position
        score = torch.empty_like(c["score"])
        score[:, :, rank_owned_position(torch.arange(score.shape[2], device="cuda"))] = c["score"]
    index_topk_decode_indices(
        score, topk, c["seq"], max_seq_len=c["msl"], query_len=c["ql"], rank_owned_order=rank_owned_order
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(topk, decode_reference(c), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["real", "nan", "ties"])
@pytest.mark.parametrize("shape", [SHAPES[0], SHAPES[7], SHAPES[15]], ids=str)
@torch.inference_mode()
def test_topk_decode_padded_outputs(kind, shape):
    """Decode indices into a view of the indexer's persistent buffer: nothing outside the view is written."""
    require_gfx950()
    c = decode_case(kind, shape)
    heads, rows, _ = c["score"].shape
    _, compared_i, where_i = outputs(heads, rows, c["nkv"], True)
    view = compared_i[0][:, where_i[0]]
    index_topk_decode_indices(c["score"], view, c["seq"], max_seq_len=c["msl"], query_len=c["ql"])
    torch.cuda.synchronize()
    tb = torch.full_like(compared_i[0], -7)
    tb[:, where_i[0]] = decode_reference(c)
    torch.testing.assert_close(compared_i[0], tb, rtol=0, atol=0)


@torch.inference_mode()
def test_topk_decode_all_padding():
    """A captured dummy batch: every request padding (seq_len 0)."""
    require_gfx950()
    reqs, ql, msl = 48, 4, 1 << 20
    score = torch.randn(1, reqs * ql, score_width(msl), device="cuda")
    c = dict(score=score, msl=msl, nblk=torch.zeros(reqs * ql, dtype=torch.int64, device="cuda"))
    seq = torch.zeros(reqs, dtype=torch.int32, device="cuda")
    topk = torch.full((1, reqs * ql, 16), -7, dtype=torch.int32, device="cuda")
    index_topk_decode_indices(score, topk, seq, max_seq_len=msl, query_len=ql)
    torch.cuda.synchronize()
    torch.testing.assert_close(topk, decode_reference(c), rtol=0, atol=0)


PREFILL_SHAPES = [
    # (q_len, seq_len) per request, max_seq_len, num_kv_heads, index heads
    ([(1000, 1000)], 1000, 1, 1),  # a fresh prompt: every row diagonal
    ([(700, 30000), (1500, 1500)], 30000, 1, 1),  # cached prefix + fresh
    ([(2000, 9000), (513, 513), (3000, 20000)], 20000, 2, 1),  # ragged, one index head shared by 2 KV heads
    ([(300, 262144), (64, 200000)], 262144, 1, 1),  # one window, 2048 blocks
    ([(129, 2049 * BLK), (40, 1 << 20)], 1 << 20, 2, 2),  # multi-window rows
    ([(1, 1), (5, 5), (128, 128), (129, 129)], 4096, 1, 1),  # tiny rows
    # 3K new on 200K-204800 prefix hits beside a fresh 3K prompt
    ([(3072, 203072), (3072, 3072), (3072, 207872)], 207872, 1, 1),
    ([(3072, 203072)], 203072, 1, 1),
    ([(64, 262000), (3072, 203072)], 200000, 1, 1),  # rows past max_seq_len
    ([(512, 1 << 20)], 1 << 20, 1, 1),  # a 1M-context chunk
]
PREFILL_IDS = [f"r{len(s[0])}-msl{s[1]}-kv{s[2]}h{s[3]}-{i}" for i, s in enumerate(PREFILL_SHAPES)]


def prefill_case(kind, shape):
    reqs, msl, nkv, heads = shape
    gen = torch.Generator().manual_seed(100 * KINDS.index(kind) + PREFILL_SHAPES.index(shape))
    # every request's chunk ends its sequence: row positions seq - q .. seq - 1
    pos = torch.cat([torch.arange(s - q, s) for q, s in reqs])
    rid = torch.cat([torch.full((q,), r) for r, (q, _) in enumerate(reqs)])
    kvl = pos + 1
    nvp = pos // BLK + 1
    seq = torch.tensor([s for _, s in reqs])
    score = make_scores(kind, heads, pos.numel(), score_width(msl), nvp, gen)
    cols = max(1, math.ceil(max(msl, int(seq.max())) / BLK))
    bt = torch.randint(0, 1 << 22, (len(reqs), cols), generator=gen)
    i32 = lambda t: t.to(torch.int32).cuda()  # noqa: E731
    return dict(
        score=score.cuda(),
        bt=i32(bt),
        seq=i32(seq),
        msl=msl,
        nvp=i32(nvp),
        rid=i32(rid),
        kvl=i32(kvl),
        nkv=nkv,
    )


def prefill_reference(c, with_table=True):
    topk = topk_reference(c["score"], c["nvp"].long(), c["msl"])
    table = (
        page_table_reference(topk, c["kvl"], c["rid"], c["bt"], c["nkv"]) if with_table else None
    )
    return topk, table


def run_prefill(c, pad=False):
    heads, rows, _ = c["score"].shape
    views, compared, where = outputs(heads, rows, c["nkv"], pad)
    index_topk_prefill(
        c["score"],
        views[0],
        c["bt"],
        c["seq"],
        views[1],
        views[2],
        max_seq_len=c["msl"],
        block_size=BLK,
        num_valid_pages=c["nvp"],
        row_req_id=c["rid"],
        kv_lens=c["kvl"],
        num_kv_heads=c["nkv"],
        pages_per_block=8,
    )
    return views, compared, where


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", PREFILL_SHAPES, ids=PREFILL_IDS)
@torch.inference_mode()
def test_topk_prefill_table(kind, shape):
    """Ragged prefill rows with the page table (FP8 main-KV attend)."""
    require_gfx950()
    c = prefill_case(kind, shape)
    _, compared, where = run_prefill(c)
    torch.cuda.synchronize()
    check_all(compared, expected(compared, where, *prefill_reference(c)))


@pytest.mark.parametrize(
    "kind,pad",
    [(k, False) for k in KINDS] + [(k, True) for k in ("real", "nan", "ties")],
    ids=[k for k in KINDS] + [f"{k}-padded" for k in ("real", "nan", "ties")],
)
@pytest.mark.parametrize("shape", PREFILL_SHAPES, ids=PREFILL_IDS)
@torch.inference_mode()
def test_topk_prefill_indices(kind, shape, pad):
    """Ragged prefill rows, indices only (windows of 4 / 8 / 16 / 20 / 26 strips), also into a persistent buffer's
    view."""
    require_gfx950()
    c = prefill_case(kind, shape)
    heads, rows, _ = c["score"].shape
    _, compared, where = outputs(heads, rows, c["nkv"], pad)
    index_topk_prefill_indices(
        c["score"],
        compared[0][:, where[0]],
        max_seq_len=c["msl"],
        block_size=BLK,
        num_valid_pages=c["nvp"],
    )
    torch.cuda.synchronize()
    tb = torch.full_like(compared[0], -7)
    tb[:, where[0]] = prefill_reference(c, False)[0]
    torch.testing.assert_close(compared[0], tb, rtol=0, atol=0)


SPLIT_KINDS = ["real", "ties", "nan", "sentinels"]


@pytest.mark.parametrize("pad", [False, True], ids=["plain", "padded"])
@pytest.mark.parametrize("kind", SPLIT_KINDS)
@pytest.mark.parametrize("shape", PREFILL_SHAPES, ids=PREFILL_IDS)
@torch.inference_mode()
def test_topk_prefill_split_launches(kind, shape, pad, monkeypatch):
    """Score tensors past the kernel's 2 GiB buffer span launch per head and per run of rows (forced with a limit of a
    third of the score): non-zero head and row starts, index heads shared by several KV heads, request ids against the
    whole block table, persistent output buffers; nothing written outside."""
    require_gfx950()
    c = prefill_case(kind, shape)
    monkeypatch.setattr(
        host, "SPAN_LIMIT", split_limit(c["score"], c["nkv"], 1, host._span_bytes(c["bt"]))
    )
    _, compared, where = run_prefill(c, pad)
    torch.cuda.synchronize()
    check_all(compared, expected(compared, where, *prefill_reference(c)))


@torch.inference_mode()
def test_topk_decode_graph_replay():
    """Captured once at max_model_len, replayed on new scores and lengths written into the captured buffers; outputs
    filled with -7 before each replay."""
    require_gfx950()
    shape = SHAPES[15]
    c = decode_case("real", shape)
    heads, rows, _ = c["score"].shape
    views, compared, where = outputs(heads, rows, c["nkv"], False)

    def call():
        index_topk_decode_indices(
            c["score"], views[0], c["seq"], max_seq_len=c["msl"], query_len=c["ql"]
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for kind in ("ties", "nan"):
        new = decode_case(kind, shape)
        for k in ("score", "bt", "seq"):
            c[k].copy_(new[k])
        for k in ("kv", "req", "nblk"):
            c[k] = new[k]
        for t in compared:
            t.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(compared[0], decode_reference(c), rtol=0, atol=0)


@torch.inference_mode()
def test_topk_deterministic():
    """Three launches of each variant give the same outputs."""
    require_gfx950()
    c = decode_case("ties", SHAPES[15])
    p = prefill_case("ties", PREFILL_SHAPES[6])
    outs = []
    for _ in range(3):
        _, compared_p, _ = run_prefill(p)
        heads, rows, _ = c["score"].shape
        ti = torch.full((heads, rows, 16), -7, dtype=torch.int32, device="cuda")
        index_topk_decode_indices(c["score"], ti, c["seq"], max_seq_len=c["msl"], query_len=c["ql"])
        hp, rp, _ = p["score"].shape
        tp = torch.full((hp, rp, 16), -7, dtype=torch.int32, device="cuda")
        index_topk_prefill_indices(
            p["score"], tp, max_seq_len=p["msl"], block_size=BLK, num_valid_pages=p["nvp"]
        )
        outs.append([*compared_p, ti, tp])
    torch.cuda.synchronize()
    for o in outs[1:]:
        assert all(torch.equal(a, b) for a, b in zip(outs[0], o))


@pytest.mark.parametrize("rows", [8, 32, 128, 512, 1024])
@torch.inference_mode()
def test_rank_owned_indices_graph_mutable_scores_and_lengths(rows):
    """One captured selector crosses both window tiers, exceptional fallbacks, padding lengths and last-chunk
    boundaries without recompiling."""
    require_gfx950()
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import rank_owned_position
    gen = torch.Generator().manual_seed(318 + rows)
    width, ql, heads = 8192, 8, 2
    score = torch.empty((heads, rows, width), device="cuda")
    storage = torch.full((heads, rows + 6, 16), -7, dtype=torch.int32, device="cuda")
    topk = storage[:, 3:3 + rows]
    seq = torch.full((rows // ql,), 1 << 20, dtype=torch.int32, device="cuda")
    pos = rank_owned_position(torch.arange(width, device="cuda"))
    score.zero_()
    call = lambda: index_topk_decode_indices(score, topk, seq, max_seq_len=1 << 20, query_len=ql, rank_owned_order=True)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for epoch, kind in enumerate(KINDS):
        blocks = torch.tensor([0, 1, 1665, 1791, 1793, 4095, 4097, 8192])
        blocks = blocks.roll(epoch).repeat((rows // ql + 7) // 8)[:rows // ql]
        lens = blocks * BLK
        seq.copy_(lens.to(device="cuda", dtype=torch.int32))
        kv = (lens.repeat_interleave(ql) - ql + torch.arange(rows) % ql + 1).clamp(min=0)
        nblk = (kv + BLK - 1) // BLK
        by_block = make_scores(kind, heads, rows, width, nblk, gen).cuda()
        score[:, :, pos] = by_block
        graph.replay()
        torch.cuda.synchronize()
        want = topk_reference(by_block, nblk.cuda(), 1 << 20)
        assert torch.equal(topk, want), (rows, epoch, kind)
        first = topk.clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(topk, first)
        assert (storage[:, :3] == -7).all() and (storage[:, -3:] == -7).all()
