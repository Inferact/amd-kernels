# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The decode indexer split over four GPUs on the rank-owned index cache (``split4.host.IndexSplit4RankOwned``):
rank r holds the blocks whose table entry p has p % 4 == r, at slot p // 4, and the block tables keep p % 4 == the
block's position % 4. The ranks' caches are sliced from a replicated one (``reference.rank_owned_pages``).

* One GPU, the four ranks in turn (peer flags pre-set, the peers' buffers local): decode scores (4 / 6 / 8 rows per
  request) on the rank-owned and the replicated cache bitwise the single-head scorer's in the rank-owned order, and
  the top-k of that order bitwise ``reference.topk_reference`` of the block order, also past one 26-strip window.
* Four GPUs over HIP IPC (``_worker``, run by ``test_split4_rank_owned_ranks4.py``): decode against the replicated
  cache and the single-head path, three runs bitwise equal, both DBO sets on two streams at once, and the decode
  captured in a CUDA graph and replayed with other lengths.
"""

import socket
import traceback

import pytest
import torch
from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.reference import (
    BLK,
    D,
    TOPK,
    cdiv,
    nvfp4_index_pages,
    rank_owned_pages,
    rank_owned_position,
    score_width,
    topk_reference,
)

from .inputs import require_gfx950

HEADS = 4
INIT_BLOCKS, LOCAL_BLOCKS = 0, 1  # the model's sparse_init_block / sparse_local_block


def _aligned_table(lens, gen, shared=False):
    """(table, num_blocks): p % 4 == position % 4 (random ids within each class), num_blocks a multiple of 4; with
    ``shared`` some requests share their first blocks with an earlier one."""
    widths = [cdiv(s, BLK) for s in lens]
    cap = max(sum((w - c + 3) // 4 for w in widths) for c in range(4)) + 2
    pools = [(torch.randperm(cap, generator=gen) * 4 + c).tolist() for c in range(4)]
    table = torch.zeros(len(lens), max(1, max(widths)), dtype=torch.int32)
    for r, w in enumerate(widths):
        for b in range(w):
            table[r, b] = pools[b % 4].pop()
    if shared:
        for r in range(1, len(lens)):
            src = 0 if r % 2 == 0 else r // 3
            n = min(widths[r], widths[src], 10 if r % 2 == 0 else 6)
            if r != src:
                table[r, :n] = table[src, :n]
    return table.cuda(), 4 * cap


def _nvfp4_pages(n, cg, zero_pages=0.0):
    """A filled NVFP4 index cache [n, 128, 72] u8; ``zero_pages``: that share of the pages holds zero keys (rows of
    equal scores)."""
    cache = torch.empty(n, BLK, 72, dtype=torch.uint8, device="cuda")
    step = 512  # pages per chunk
    for p0 in range(0, n, step):
        m = min(step, n - p0)
        keys = torch.randn(m * BLK, D, device="cuda", generator=cg) * 2
        if zero_pages:
            z = torch.rand(m, device="cuda", generator=cg) < zero_pages
            keys.view(m, BLK, D)[z] = 0
        cache[p0 : p0 + m] = nvfp4_index_pages(keys.bfloat16())
    return cache


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def _decode_inputs(lens, ql, seed, shared=True):
    gen = torch.Generator().manual_seed(seed)
    cg = torch.Generator(device="cuda").manual_seed(seed)
    table, num_blocks = _aligned_table(lens, gen, shared)
    cache = _nvfp4_pages(num_blocks, cg)
    q = (torch.randn(len(lens) * ql, HEADS, D, device="cuda", generator=cg) * 0.5).to(torch.float8_e4m3fn)
    return table, cache, q, torch.tensor(lens, dtype=torch.int32, device="cuda")


def _single_head_scores(q, cache, table, seq, ql, init_blocks, local_blocks):
    """The per-head decode scores [4, rows, W] (block order, NaN unwritten)."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.score_decode import host

    max_seq = max(int(seq.max()), 1)
    out = torch.full((HEADS, q.shape[0], score_width(max_seq)), float("nan"), device="cuda")
    for h in range(HEADS):
        host.nvfp4_index_score_decode(
            q[:, h : h + 1], cache, out[h : h + 1], table, seq, init_blocks, local_blocks, ql, max_seq
        )
    return out


def _push_sim(rank, q, cache, table, seq, out, ql, sharded, init_blocks, local_blocks, grid=0, topk_idx=None,
              max_seq_len=0):
    """Rank ``rank``'s fused push on one GPU: the heads' destinations are slices of ``out`` [4, rows, W], the peers'
    index-q rows sit in a local receive buffer, their flag words pre-set. ``topk_idx``: the fused top-k too (of
    ``out[rank]`` as it stands, so the peers' columns must be in already)."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import host
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import kernel as K

    rows, width = out.shape[1], out.shape[2]
    counts = torch.zeros(K.CNT_WORDS, dtype=torch.int32, device="cuda")
    flags = torch.zeros(K.FLAG_BYTES // 4, dtype=torch.int32, device="cuda")
    flags[K.SC_FLAGS : K.SC_FLAGS + 4] = 1 << 30
    flags[K.SC_FLAGS + rank] = 0
    iq_buf = torch.full((HEADS, max(rows, 1), D), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    for p in range(HEADS):
        if p != rank:
            iq_buf[p, :rows] = q[:, p]
            w = K.SC_IQ + p * K.SC_MAX_REQS
            flags[w : w + K.SC_MAX_REQS] = 1 << 30
    host.nvfp4_index_score_decode_rank_owned_push(
        q[:, rank].contiguous(), cache, [out[h].data_ptr() for h in range(HEADS)], width, rows * width * 4, table,
        seq, init_blocks, local_blocks, rank, counts, [flags.data_ptr()] * 4, [iq_buf.data_ptr()] * 4, max(rows, 1),
        grid=grid, query_len=ql, sharded=sharded, topk_idx=topk_idx, max_seq_len=max_seq_len,
    )
    torch.accelerator.synchronize()


def _topk_ref(score, seq, ql):
    """``reference.topk_reference`` of [1, rows, W] block-order decode scores at max(seq_lens): topk [rows, 16]."""
    rows = score.shape[1]
    kv = (seq.long().repeat_interleave(ql) - ql + torch.arange(rows, device=score.device) % ql + 1).clamp(min=0)
    return topk_reference(score, (kv + BLK - 1) // BLK, max(int(seq.max()), 1))[0]


def _topk_decode_indices(score, seq, ql, rank_owned_order, max_seq_len=None):
    """The indices-only decode top-k of [1, rows, W] scores: topk [rows, 16]."""
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.topk import index_topk_decode_indices

    topk = torch.full((1, score.shape[1], TOPK), -7, dtype=torch.int32, device="cuda")
    index_topk_decode_indices(
        score, topk, seq, max_seq_len=max_seq_len or max(int(seq.max()), 1), query_len=ql, block_size=BLK,
        rank_owned_order=rank_owned_order,
    )
    return topk[0]


def _check_decode(lens, ql, seed, init_blocks, local_blocks, grid=0, shared=True):
    table, cache, q, seq = _decode_inputs(lens, ql, seed, shared)
    rows = len(lens) * ql
    expected = _single_head_scores(q, cache, table, seq, ql, init_blocks, local_blocks)
    width = expected.shape[2]
    blocks = torch.arange(width, device="cuda")
    pos = rank_owned_position(blocks)
    nblk = ((seq.long() + BLK - 1) // BLK).repeat_interleave(ql)[:, None]
    # a rank writes a run (its 16 blocks of a chunk) iff 64 c + rank < nblk
    in_run = (blocks // 64 * 64 + blocks % 4)[None] < nblk
    inside = blocks[None] < nblk
    outs = {}
    for sharded in (False, True):
        runs = []
        for _ in range(2):
            out = torch.full((HEADS, rows, width), float("nan"), device="cuda")
            for rank in range(4):
                c = rank_owned_pages(cache, rank) if sharded else cache
                _push_sim(rank, q, c, table, seq, out, ql, sharded, init_blocks, local_blocks, grid)
            runs.append(out)
        assert torch.equal(runs[0].view(torch.int32), runs[1].view(torch.int32))
        out = runs[0]
        by_block = out[:, :, pos]  # [h, row, b]: block b's value
        assert torch.equal(by_block[:, inside].view(torch.int32), expected[:, inside].view(torch.int32)), sharded
        assert (by_block[:, in_run & ~inside] == float("-inf")).all(), sharded
        assert by_block[:, ~in_run].isnan().all(), sharded
        outs[sharded] = out
    assert torch.equal(outs[False].view(torch.int32), outs[True].view(torch.int32))
    for h in range(HEADS):
        want = _topk_ref(expected[h : h + 1], seq, ql)
        want_idx = _topk_decode_indices(expected[h : h + 1], seq, ql, False)
        got_idx = _topk_decode_indices(outs[True][h : h + 1], seq, ql, True)
        assert torch.equal(want_idx, want) and torch.equal(got_idx, want), h
    # the fused top-k (``score_topk``): every rank's head, over the peers'
    # scores pushed above and its own push, whose slots of head ``rank`` start
    # as MARKER (the peers' have arrived, this rank's have not); its push
    # rewrites the same bits into the other heads, and its selectors leave
    # MARKER over every chunk of their rows' requests in head ``rank``, nothing
    # else changed (max_seq_len: the longest row's, and the most the score
    # width holds)
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4.decode_score import MARKER

    req_chunks = ((seq.long() + BLK - 1) // BLK + 63) // 64 * 64
    cols = torch.arange(width, device="cuda")
    reset = cols[None] < req_chunks.repeat_interleave(ql)[:, None]  # [rows, W]
    full = outs[True].clone()
    for msl in (max(int(seq.max()), 1), width * BLK):
        for rank in range(4):
            tk = torch.full((1, rows, TOPK), -7, dtype=torch.int32, device="cuda")
            outs[True].copy_(full)
            outs[True].view(torch.int32)[rank][reset & (cols % 64 // 16 == rank)[None]] = MARKER
            _push_sim(
                rank, q, rank_owned_pages(cache, rank), table, seq, outs[True], ql, True, init_blocks, local_blocks,
                min(grid or 512, 128), topk_idx=tk, max_seq_len=msl,
            )
            want_buf = full.clone().view(torch.int32)
            want_buf[rank][reset] = MARKER
            assert torch.equal(want_buf, outs[True].view(torch.int32)), (rank, msl)
            want = _topk_decode_indices(expected[rank : rank + 1], seq, ql, False, msl)
            assert torch.equal(tk[0], want), (rank, msl)
    outs[True].copy_(full)


@pytest.mark.parametrize("ql", [4, 6, 8])
@pytest.mark.parametrize("num_reqs", [1, 5, 16, 17, 45])
@pytest.mark.parametrize(("init_blocks", "local_blocks"), [(0, 1), (1, 2)])
@torch.inference_mode()
def test_decode_rank_owned_matches_single_head(ql, num_reqs, init_blocks, local_blocks):
    """The fused decode push on the replicated and the rank-owned cache: per block bitwise the single-head scorer,
    -inf past the block count inside a written run, nothing else written; both caches' pushes bitwise equal; two
    runs equal; the rank-owned-order top-k equals the block-order top-k on the single-head scores."""
    require_gfx950()
    gen = torch.Generator().manual_seed(num_reqs * 10 + ql)
    lens = torch.randint(1, 40000, (num_reqs,), generator=gen).tolist()
    special = [0, 1, 2, 3, 127, 128, 129, 1024, 64 * 128 + 3, 256 * 128]
    for i, v in enumerate(special[: (num_reqs + 1) // 2]):
        lens[(2 * i + 1) % num_reqs] = v
    _check_decode(lens, ql, num_reqs, init_blocks, local_blocks)


@pytest.mark.parametrize("ql", [4, 6, 8])
@torch.inference_mode()
def test_decode_rank_owned_small_grid_and_padding(ql):
    """Grid 3 (many runs per workgroup, several 64-page windows per wave) and a batch of padding requests only."""
    require_gfx950()
    gen = torch.Generator().manual_seed(77)
    lens = torch.randint(1, 30000, (20,), generator=gen).tolist()
    _check_decode(lens, ql, 3, 0, 1, grid=3)
    _check_decode([0] * 5, ql, 4, 0, 1, grid=3)


@pytest.mark.parametrize("ql", [4, 8])
@pytest.mark.parametrize("strips", [26, 28, 64])
@torch.inference_mode()
def test_decode_indices_rank_owned_order_past_one_window(ql, strips):
    """The indices-only decode top-k on rank-owned-order rows longer than its 26-strip window (1664 blocks): a live
    block of the last 64-block chunk can sit past the row's block count (block 64 c + 4 j + r at 64 c + 16 r + j),
    and a later window must still read it. Every row's best blocks are put there; the result equals the block-order
    top-k and ``reference.topk_reference``."""
    require_gfx950()
    window = strips * 64
    nblks = [window - 5, window]
    for base in (window, min(2 * window + 64, 8192 - 64)):
        nblks += [base + rem for rem in (1, 2, 3, 17, 33, 47, 61, 63, 64)]
    seq = torch.tensor([n * BLK for n in nblks], dtype=torch.int32, device="cuda")
    width = score_width(int(seq.max()))
    rows = len(nblks) * ql
    cg = torch.Generator(device="cuda").manual_seed(ql)
    blk = torch.randn((1, rows, width), device="cuda", generator=cg)
    for i, n in enumerate(nblks):
        tail = torch.arange(n // 64 * 64, n, device="cuda")
        past = tail[rank_owned_position(tail) >= n][:TOPK]
        blk[0, i * ql : (i + 1) * ql, past] = 100.0 + torch.arange(past.numel(), device="cuda", dtype=torch.float32)
    cp = torch.empty_like(blk)
    cp[:, :, rank_owned_position(torch.arange(width, device="cuda"))] = blk
    want = _topk_decode_indices(blk, seq, ql, False)
    got = _topk_decode_indices(cp, seq, ql, True)
    assert torch.equal(got, want)
    # decode rows see their whole sequence: kv_len = seq_len for the block count
    nblk_rows = torch.tensor(nblks, device="cuda").repeat_interleave(ql)
    assert torch.equal(got, topk_reference(blk, nblk_rows, int(seq.max()))[0])


@torch.inference_mode()
def test_decode_rank_owned_long_rows():
    """The fused decode and its top-k on requests of 150K-300K tokens (one or two top-k windows)."""
    require_gfx950()
    _check_decode([150000, 215000, 260000, 300000], 4, 5, 0, 1)


# ---------------------------------------------------------------------------
# four GPUs
# ---------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, port, errfile):
    try:
        with torch.inference_mode():
            _worker_body(rank, port)
    except BaseException:
        with open(f"{errfile}.{rank}", "w") as f:
            f.write(traceback.format_exc())
        raise


def _worker_body(rank, port):
    import torch.distributed as dist
    from amd_kernels.indexer.indexer_nvfp4_d128_p128_top16.split4 import host

    torch.accelerator.set_device_index(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4)
    group = dist.group.WORLD
    dev = torch.device("cuda", rank)
    width = score_width(1 << 20)
    cps = {ql: [host.IndexSplit4RankOwned(rank, group, dev, 64 * ql, width, ql) for _ in range(2)] for ql in (4, 6, 8)}
    host.init_rank_owned(rank, 4, group, dev, width, 64, 4, 2)
    assert host.get_rank_owned(1) is not host.get_rank_owned(0)
    failures = []

    def check(cond, what):
        if not cond:
            failures.append(what)

    # -- decode: rank-owned vs replicated vs the single-head path (eager, repeat)
    # 45 requests: runs per workgroup; 3 (<= 64 runs): the unit split
    for ql, nreq in ((4, 45), (6, 45), (8, 45), (4, 3), (8, 3)):
        gen = torch.Generator().manual_seed(ql + nreq)
        lens = torch.randint(30000, 200001, (nreq,), generator=gen).tolist()
        lens[-1] = 0  # a graph padding request
        if nreq == 45:
            lens[-3] = 0
        else:
            lens[0] = 300001  # past the top-k's first 1664-block window
            lens[1] = min(lens[1], 150000)  # <= 64 runs per rank: the unit split
        table, cache, q, seq = _decode_inputs(lens, ql, seed=11 + ql, shared=False)
        shard = rank_owned_pages(cache, rank)
        iq = q[:, rank].contiguous()
        expected = _single_head_scores(q, cache, table, seq, ql, INIT_BLOCKS, LOCAL_BLOCKS)
        ref = _topk_ref(expected[rank : rank + 1], seq, ql)
        for i in range(3):
            for kv, sharded in ((cache, False), (shard, True)):
                score = cps[ql][0].score_push(iq, kv, table, seq, INIT_BLOCKS, LOCAL_BLOCKS, width, sharded=sharded)
                idx = _topk_decode_indices(score, seq, ql, True)
                check(torch.equal(idx, ref), ("decode indices", ql, nreq, sharded, i))
                # the scorer with the top-k fused (one launch)
                fused = torch.full((1, nreq * ql, TOPK), -7, dtype=torch.int32, device="cuda")
                cps[ql][1].score_topk(
                    iq, kv, table, seq, INIT_BLOCKS, LOCAL_BLOCKS, width, fused,
                    max(int(seq.max()), 1) if i < 2 else 1 << 20, sharded=sharded,
                )
                check(torch.equal(fused[0], ref), ("decode fused", ql, nreq, sharded, i))

    # -- DBO: both sets at once on two streams --------------------------------
    scp = host.get_rank_owned(0), host.get_rank_owned(1)
    lens = torch.randint(30000, 200001, (30,), generator=torch.Generator().manual_seed(8)).tolist()
    table, cache, q, seq = _decode_inputs(lens, 4, seed=31, shared=False)
    shard = rank_owned_pages(cache, rank)
    iq = q[:, rank].contiguous()
    ex = _single_head_scores(q, cache, table, seq, 4, INIT_BLOCKS, LOCAL_BLOCKS)
    ref_a = _topk_ref(ex[rank : rank + 1], seq, 4)
    seq_b = seq[:20].clone()
    ex_b = _single_head_scores(q[: 20 * 4], cache, table[:20], seq_b, 4, INIT_BLOCKS, LOCAL_BLOCKS)
    ref_b = _topk_ref(ex_b[rank : rank + 1], seq_b, 4)
    # 8 requests (32 rows): ``score_topk``'s one-launch path
    seq_c = seq[:8].clone()
    ex_c = _single_head_scores(q[: 8 * 4], cache, table[:8], seq_c, 4, INIT_BLOCKS, LOCAL_BLOCKS)
    ref_c = _topk_ref(ex_c[rank : rank + 1], seq_c, 4)
    s0, s1 = torch.cuda.Stream(), torch.cuda.Stream()
    torch.accelerator.synchronize()
    for dbo_rep in range(6):
        # ``score_topk`` on both sets (one launch each: 120 and 32 rows)
        fused = dbo_rep in (1, 3, 4, 5)
        fa = torch.full((1, 30 * 4, TOPK), -7, dtype=torch.int32, device="cuda")
        fb = torch.full((1, 8 * 4, TOPK), -7, dtype=torch.int32, device="cuda")
        with torch.cuda.stream(s0):
            if fused:
                scp[0].score_topk(iq, shard, table, seq, INIT_BLOCKS, LOCAL_BLOCKS, width, fa, 1 << 20)
            else:
                sa = scp[0].score_push(iq, shard, table, seq, INIT_BLOCKS, LOCAL_BLOCKS, width, sharded=True)
                da = _topk_decode_indices(sa, seq, 4, True)
        with torch.cuda.stream(s1):
            if fused:
                scp[1].score_topk(iq[: 8 * 4], shard, table[:8], seq_c, INIT_BLOCKS, LOCAL_BLOCKS, width, fb, 1 << 20)
            else:
                sb = scp[1].score_push(
                    iq[: 20 * 4], shard, table[:20], seq_b, INIT_BLOCKS, LOCAL_BLOCKS, width, sharded=True
                )
                db = _topk_decode_indices(sb, seq_b, 4, True)
        torch.accelerator.synchronize()
        if fused:
            check(torch.equal(fa[0], ref_a), ("dbo", "decode a fused"))
            check(torch.equal(fb[0], ref_c), ("dbo", "decode c fused"))
            da, db = ref_a, ref_b
        check(torch.equal(da, ref_a), ("dbo", "decode a"))
        check(torch.equal(db, ref_b), ("dbo", "decode b"))

    # -- graphs: the decode (scorer + top-k, and the fused one) captured at 24
    # requests, replayed with other lengths (padding requests, all padding) ----
    msl = 1 << 18  # the captured max_seq_len
    # (3 and 4 requests: ``score_topk``'s one-launch path, <= 32 rows)
    for ql, n_req in ((4, 24), (6, 24), (8, 24), (4, 3), (8, 4)):
        cp = cps[ql][0]
        gen = torch.Generator().manual_seed(60 + ql)
        lens = torch.randint(30000, 150000, (n_req,), generator=gen).tolist()
        table, cache, q, seq = _decode_inputs(lens, ql, seed=60 + ql, shared=False)
        shard = rank_owned_pages(cache, rank)
        iq = q[:, rank].contiguous()
        rows = n_req * ql

        out_f = torch.full((1, rows, TOPK), -7, dtype=torch.int32, device="cuda")

        def fwd():
            s = cp.score_push(iq, shard, table, seq, INIT_BLOCKS, LOCAL_BLOCKS, width)
            res = _topk_decode_indices(s, seq, ql, True, msl)
            cp.score_topk(iq, shard, table, seq, INIT_BLOCKS, LOCAL_BLOCKS, width, out_f, msl)
            return res

        fwd()  # compiles outside the capture
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_idx = fwd()
        torch.accelerator.synchronize()
        # k 3: two short requests, few enough runs for the scorer's unit split
        # (chosen on the device: the graph captured whole runs)
        for k in range(4):
            new = torch.tensor(lens, dtype=torch.int32)
            if k == 2:
                new.fill_(0)
            elif k == 3:
                new[2:] = 0
                new[:2] = torch.tensor([20000, 70001])
            else:
                new = (new.float() * torch.rand(new.shape, generator=gen)).to(torch.int32)
                new[::5] = 0
            seq.copy_(new)
            for t in (out_idx, out_f):
                t.fill_(-7)
            torch.accelerator.synchronize()
            dist.barrier()
            graph.replay()
            torch.accelerator.synchronize()
            ex = _single_head_scores(q, cache, table, seq, ql, INIT_BLOCKS, LOCAL_BLOCKS)
            # the block-order top-k at the replayed lengths' own max_seq_len (the
            # single-head scores are as wide as that needs)
            want = _topk_ref(ex[rank : rank + 1], seq, ql)
            check(torch.equal(out_idx[:rows], want), ("graph indices", ql, k))
            check(torch.equal(out_f[0], want), ("graph fused", ql, k))
        dist.barrier()

    dist.barrier()
    if failures:
        raise AssertionError(f"rank {rank}: {failures[:12]}")
    dist.destroy_process_group()


@torch.inference_mode()
def test_fused_topk_waiter_guard_counts_workgroups_per_row():
    """Four cooperating waves make 136 rows mean 136 waiting workgroups."""
    require_gfx950()
    ql, rows, ctx = 8, 136, 256
    table, cache, q, seq = _decode_inputs([ctx] * (rows // ql), ql, 33, False)
    out = torch.empty((4, rows, score_width(ctx)), dtype=torch.float32, device="cuda")
    topk = torch.empty((1, rows, TOPK), dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match=str(rows)):
        _push_sim(0, q, rank_owned_pages(cache, 0), table, seq, out, ql, True,
                  0, 1, 512, topk_idx=topk, max_seq_len=ctx)
