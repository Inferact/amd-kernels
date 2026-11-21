# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router split-K partials [KS, M, 896] -> AITER's biased top-16 (``topk_reg_kernel``: sigmoid, + correction bias, one
group, renormalised), one wave per token (gfx950): its ids, slot order, tie rule and weights bit for bit.

The wave sums the KS slabs of its token in slice order, so the logits are the same bits whichever slice finished last.
Lane l holds the experts 256 q + 4 l + j (q 0-3, j 0-3; q 3 only on lanes 0-31), AITER's float4 order, so every slab
load is one coalesced 16-byte read and AITER's tie rank (the lane's DPP-tree priority, then 4 q + j) is a function of
(l, 4 q + j). The selection is the top 16 of the total order (biased score descending, tie rank ascending):

1. pivot: the 16th largest of the 64 lane maxima (each lane counts the maxima above its own through LDS) is at most the
   16th largest key, so the candidates are the keys at or above it, 16 to a few dozen of 896;
2. the candidates go to LDS in lane order; each lane ranks one against all the others (key, then tie rank), the 16
   best scatter to their slots, and the 16 unbiased sigmoids are summed in slot order, divided and scaled.
3. with more than 64 candidates (a plateau of exact ties) the slow path takes over: each lane sorts its 16 keys, 16 rounds
   pop the wave-wide maximum until 16 are consumed, and the ties at the 16th key are chosen by tie rank.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

E, TOPK, NQ, EPL = 896, 16, 4, 16
RANK_UNROLL = 32  # candidates ranked per loop step (all loads first: the step costs one LDS latency)
MAXC = 64  # candidates the fast path ranks; one LDS slot each
SENTINELS = RANK_UNROLL  # slots after the candidates holding key 0, so a rank step never reads past them
DUMMY0 = MAXC + SENTINELS  # the slots the unselected elements write to: one per lane (a slot shared by the lanes would
ENT_SLOTS = DUMMY0 + 64  # serialise every store on the LDS bank)
ENT_WORDS = ENT_SLOTS * 4  # LDS words per wave: per slot (tie term, key, sigmoid bits, expert)
LDS_WORDS = {"ent": ENT_WORDS, "osig": TOPK + 1, "oid": TOPK + 1, "tie": 64, "tex": 64, "tmask": 64}
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}
DPP_TREE = (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143)
NEG_LOG2E = -1.44269504088896340736


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp(v, ctrl, row_mask=0xF):
    """DPP move; a lane without a source reads 0 (the identity of the unsigned max and of the sum it feeds)."""
    return fx.Uint32(_rocdl_ops.update_dpp(T.i32, _ir(fx.Int32(0)), _ir(v), ctrl, row_mask, 0xF, True))


def _scan(x):
    """Wave inclusive prefix sum of int32 per lane (AITER's DPP scan); lane 63 holds the total."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        x = x + fx.Int32(_dpp(x, ctrl))
    x = x + fx.Int32(_dpp(x, 0x142, 0xA))
    return x + fx.Int32(_dpp(x, 0x143, 0xC))


def _batcher_pairs(n):
    """Odd-even merge sort comparators (i < j, ascending) for any n."""
    pairs, p = [], 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


SORT_PAIRS = _batcher_pairs(EPL)


def _ballot(pred):
    return fx.Int64(fx.rocdl.ballot(T.i64, pred))


def _popcount(mask):
    return fx.Int32(fx.math.ctpop(mask).to(fx.Int32))


def _lane_rank(lane):
    """AITER's legacy wave64 DPP tie priority of a lane (a permutation of 0-63; lower wins)."""
    low2 = (lane & 3) ^ (((lane >> 2) & 1) * 3)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    return ((~lane) & 0x38) | bit2 | low2



from types import SimpleNamespace


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _ir(fx.Int64(byte_addr)))


def topk_q_scores(pr, br, lane, tok, i32_m, ks, slab_cm, q):
    """Group ``q`` (an int or a wave-uniform Int32: the experts 256 q + 4 lane + j): the ``ks`` slab vectors summed in slice
    order, then ``topk_q_keys``. Returns [(sigmoid, key)] for j = 0..3."""
    dynamic = not isinstance(q, int)
    valid = ((q != NQ - 1) | (lane < 32)) if dynamic else ((lane < 32) if q == NQ - 1 else None)
    parts = [Vec(buffer_ops.buffer_load(pr, (s * i32_m + tok) * E + 256 * q + 4 * lane, vec_width=4, dtype=fx.Float32,
                                        mask=valid, cache_modifier=slab_cm)) for s in range_constexpr(ks)]
    acc = parts[0]
    for s in range_constexpr(1, ks):
        acc = acc + parts[s]
    return topk_q_keys(acc, br, lane, q)


def topk_q_keys(acc, br, lane, q, bv=None):
    """Group ``q``'s summed logits ``acc`` (a vector of 4 fp32 per lane): AITER's sigmoid (rcp(1 + exp2(-log2(e) x)), the
    product in double), the biased score and its order-preserving unsigned key. Returns [(sigmoid, key)] for j = 0..3."""
    dynamic = not isinstance(q, int)
    valid = ((q != NQ - 1) | (lane < 32)) if dynamic else ((lane < 32) if q == NQ - 1 else None)
    if bv is None:
        bv = Vec(buffer_ops.buffer_load(br, 256 * q + 4 * lane, vec_width=4, dtype=fx.Float32, mask=valid))
    out = []
    for j in range_constexpr(4):
        y = (fx.Float32(acc[j]).to(fx.Float64) * fx.Float64(NEG_LOG2E)).to(fx.Float32)
        y = (y > -126.0).select(y, fx.Float32(-126.0))  # exp2f's denormal range only ever rounds 1 + e to 1
        e = fx.Float32(_rocdl_ops.exp2(T.f32, _ir(y)))
        sg = fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(1.0) + e))
        u = (sg + fx.Float32(bv[j])).bitcast(fx.Int32)
        k = fx.Uint32(u ^ ((u >> 31) | fx.Int32(-(2**31))))
        out.append((sg, k if valid is None else valid.select(k, fx.Uint32(0))))
    return out


def topk_new(lane):
    return SimpleNamespace(sig=[], key=[], rl=_lane_rank(lane))


def topk_pivot(st, smem, wv, lane):
    """The pivot: the smallest lane maximum among those with at most 3 larger ones in their row of 16 lanes. Every row
    contributes at least 4 distinct keys at or above it, so at least 16 keys are, and the 16th largest key is at or
    above it: the candidates are the keys at or above the pivot (``st.nc``, 16 to a few dozen of 896)."""
    key = st.key
    head = key[0]
    for i in range_constexpr(1, EPL):
        head = (key[i] > head).select(key[i], head)
    above = fx.Int32(0)
    for k in range_constexpr(1, 16):
        above = above + (_dpp(head, 0x120 + k) > head).select(fx.Int32(1), fx.Int32(0))
    inv = (above <= 3).select(~head, fx.Uint32(0))
    for ctrl in DPP_TREE:
        r = _dpp(inv, ctrl)
        inv = (r > inv).select(r, inv)
    pivot = ~fx.Uint32(fx.rocdl.readlane(T.i32, inv, fx.Int32(63)))
    cand = [key[i] >= pivot for i in range_constexpr(EPL)]
    ncl = cand[0].select(fx.Int32(1), fx.Int32(0))
    for i in range_constexpr(1, EPL):
        ncl = ncl + cand[i].select(fx.Int32(1), fx.Int32(0))
    incl = _scan(ncl)
    st.cand = cand
    st.excl = incl - ncl
    st.nc = fx.Int32(fx.rocdl.readlane(T.i32, incl, fx.Int32(63)))


def _ent_addr(smem, wv, slot):
    return fx.Int32(fx.ptrtoint(smem.ent.ptr)) + (wv * ENT_SLOTS + slot) * 16


def _entry(st_key, tie_rank, sig, expert):
    """A candidate's LDS entry (16 bytes): the 64-bit order key (the score key above 0xFFFF - tie rank, so a larger
    key is better), the sigmoid bits and the expert id."""
    return Vec.from_elements([fx.Int32(0xFFFF) - tie_rank, fx.Int32(st_key), sig.bitcast(fx.Int32), expert], fx.Int32)


def topk_fast_write(st, smem, wv, lane):
    """The candidates to LDS in expert-lane order (slots 0..nc-1; the others go to the spare slots)."""
    run = st.excl
    for i in range_constexpr(EPL):
        slot = st.cand[i].select(run, fx.Int32(DUMMY0) + lane)
        ent = _entry(st.key[i], (st.rl << 5) | i, st.sig[i], 256 * (i // 4) + 4 * lane + (i % 4))
        llvm.store(_ir(ent), _lds_ptr(_ent_addr(smem, wv, slot)), alignment=16)
        run = run + st.cand[i].select(fx.Int32(1), fx.Int32(0))
    zero = fx.Int32(0)
    llvm.store(_ir(Vec.from_elements([zero, zero, zero, zero], fx.Int32)),
               _lds_ptr(_ent_addr(smem, wv, (lane < SENTINELS).select(st.nc + lane, fx.Int32(DUMMY0) + lane))), alignment=16)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)


def topk_rank_start(st, smem, wv, lane):
    """The lane's own candidate (slot ``lane``)."""
    own = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(_ent_addr(smem, wv, lane)), alignment=16))
    st.mykey = fx.Uint64(llvm.bitcast(T.i64, _ir(Vec.from_elements([own[0], own[1]], fx.Int32))))
    st.mysig = fx.Int32(own[2])
    st.myid = fx.Int32(own[3])


def topk_rank_step(rank, e, st, smem, wv):
    """Sixteen more candidates (slots e..e+15) ranked against the lane's own: ``rank`` counts the better ones."""
    others = [fx.Uint64(llvm.load(T.i64, _lds_ptr(_ent_addr(smem, wv, e + u)), alignment=8))
              for u in range_constexpr(RANK_UNROLL)]
    for u in range_constexpr(RANK_UNROLL):
        rank = rank + (others[u] > st.mykey).select(fx.Int32(1), fx.Int32(0))
    return rank


def topk_slow_select(st):
    """The slow path (more than MAXC candidates): each lane sorts its keys, 16 rounds pop the wave-wide maximum until 16
    are consumed (the last popped key is the 16th largest, and the keys consumed before its value first came up are
    the larger ones), then the census of the exact ties at that key."""
    key = st.key
    srt = list(key)
    for i, j in SORT_PAIRS:
        lt = srt[i] < srt[j]
        srt[i], srt[j] = lt.select(srt[i], srt[j]), lt.select(srt[j], srt[i])
    consumed = fx.Int32(0)
    cand = fx.Uint32(0)
    before = fx.Int32(0)
    found = fx.Int32(0)
    prev = fx.Uint32(0)
    start = fx.Int32(0)
    for _ in range_constexpr(TOPK):
        head = srt[EPL - 1]
        m = head
        for ctrl in DPP_TREE:
            r = _dpp(m, ctrl)
            m = (r > m).select(r, m)
        gmax = fx.Uint32(fx.rocdl.readlane(T.i32, m, fx.Int32(63)))
        at = head == gmax
        n = _popcount(_ballot(at))
        for t in range_constexpr(EPL - 1, 0, -1):
            srt[t] = at.select(srt[t - 1], srt[t])
        srt[0] = at.select(fx.Uint32(0), srt[0])
        start = (gmax != prev).select(consumed, start)
        prev = gmax
        hit = (found == 0) & (consumed + n >= TOPK)
        cand = hit.select(gmax, cand)
        before = hit.select(start, before)
        found = hit.select(fx.Int32(1), found)
        consumed = consumed + n

    need = fx.Int32(TOPK) - before
    strict = [key[i] > cand for i in range_constexpr(EPL)]
    tie = [key[i] == cand for i in range_constexpr(EPL)]
    nties = tie[0].select(fx.Int32(1), fx.Int32(0))
    for i in range_constexpr(1, EPL):
        nties = nties + tie[i].select(fx.Int32(1), fx.Int32(0))
    total_ties = fx.Int32(fx.rocdl.readlane(T.i32, _scan(nties), fx.Int32(63)))
    return SimpleNamespace(sig=st.sig, key=key, rl=st.rl, need=need, tie=tie, strict=strict, nties=nties,
                           total_ties=total_ties)


def topk_tie_masks(st, smem, wv, lane):
    """Slow path, only with more exact ties than slots: the ties with the lowest tie ranks, counted over the lanes in
    tie-rank order then 4 q + j, as a bit mask per lane in LDS."""
    smem.tie.ptr[wv * 64 + st.rl] = st.nties
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)
    at_rank = fx.Int32(smem.tie.ptr[wv * 64 + lane])
    smem.tex.ptr[wv * 64 + lane] = _scan(at_rank) - at_rank
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)
    run = fx.Int32(smem.tex.ptr[wv * 64 + st.rl])
    bits = fx.Int32(0)
    for i in range_constexpr(EPL):
        take = st.tie[i] & (run < st.need)
        bits = bits | take.select(fx.Int32(1 << i), fx.Int32(0))
        run = run + st.tie[i].select(fx.Int32(1), fx.Int32(0))
    smem.tmask.ptr[wv * 64 + lane] = bits


def topk_slow_out(st, smem, wv, lane):
    """Slow path: select (the larger keys and the ties chosen), rank the 16 selected in LDS and scatter them to their
    slots."""
    sig, key, strict, tie, need, total_ties, rl = st.sig, st.key, st.strict, st.tie, st.need, st.total_ties, st.rl
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)
    slow_bits = fx.Int32(smem.tmask.ptr[wv * 64 + lane])
    exact = total_ties == need
    sel = [strict[i] | (exact.select(tie[i], ((slow_bits >> i) & 1) != 0)) for i in range_constexpr(EPL)]

    nsel = sel[0].select(fx.Int32(1), fx.Int32(0))
    for i in range_constexpr(1, EPL):
        nsel = nsel + sel[i].select(fx.Int32(1), fx.Int32(0))
    run = _scan(nsel) - nsel
    for i in range_constexpr(EPL):
        slot_i = sel[i].select(run, fx.Int32(DUMMY0) + lane)
        ent = _entry(key[i], (rl << 5) | i, sig[i], 256 * (i // 4) + 4 * lane + (i % 4))
        llvm.store(_ir(ent), _lds_ptr(_ent_addr(smem, wv, slot_i)), alignment=16)
        run = run + sel[i].select(fx.Int32(1), fx.Int32(0))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)

    own = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(_ent_addr(smem, wv, lane % TOPK)), alignment=16))
    mykey = fx.Uint64(llvm.bitcast(T.i64, _ir(Vec.from_elements([own[0], own[1]], fx.Int32))))
    rank = fx.Int32(0)
    for e in range_constexpr(TOPK):
        other = fx.Uint64(llvm.load(T.i64, _lds_ptr(_ent_addr(smem, wv, e)), alignment=8))
        rank = rank + (other > mykey).select(fx.Int32(1), fx.Int32(0))
    return rank, fx.Int32(own[2]), fx.Int32(own[3])


def topk_finish(rank, valid, mysig, myid, lane, smem, wv, wr, ir, tok, i32_os, i32_scale):
    """The ``valid`` lanes hold distinct ranks 0..: each hands its sigmoid and expert id to LDS slot ``rank``; lanes 0-15
    read their slot, the sum of the 16 sigmoids runs in slot order (lane by lane), and lanes 0-15 write
    ``sigmoid * (scale / sum)``. Returns (expert id, weight) of the lane."""
    base = wv * (TOPK + 1)
    slot = ((rank < TOPK) & valid).select(rank, fx.Int32(TOPK))
    smem.osig.ptr[base + slot] = mysig.bitcast(fx.Float32)
    smem.oid.ptr[base + slot] = myid
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)
    out_sig = fx.Float32(smem.osig.ptr[base + lane % TOPK])
    out_id = fx.Int32(smem.oid.ptr[base + lane % TOPK])
    bits = out_sig.bitcast(fx.Int32)
    total = fx.Int32(fx.rocdl.readlane(T.i32, bits, fx.Int32(0))).bitcast(fx.Float32)
    for r in range_constexpr(1, TOPK):
        total = total + fx.Int32(fx.rocdl.readlane(T.i32, bits, fx.Int32(r))).bitcast(fx.Float32)
    factor = i32_scale.bitcast(fx.Float32) / total
    w = (total > 0.0).select(out_sig * factor, fx.Float32(0.0))
    keep = lane < TOPK
    buffer_ops.buffer_store(w, wr, tok * i32_os + lane, mask=keep)
    buffer_ops.buffer_store(out_id, ir, tok * i32_os + lane, mask=keep)
    return out_id, w


@launcher
def compile_router_sigmoid_bias_top16_e896(ks: int):
    """The top-16 launch alone: ``ks`` slabs (1: fp32 logits) in, one wave per token."""
    name = f"router_sigmoid_bias_top16_e896_ks{ks}"

    @fx.struct
    class Shared:
        ent: fx.Array[fx.Int32, LDS_WORDS["ent"], 16]
        osig: fx.Array[fx.Float32, LDS_WORDS["osig"], 16]
        oid: fx.Array[fx.Int32, LDS_WORDS["oid"], 16]
        tie: fx.Array[fx.Int32, LDS_WORDS["tie"], 16]
        tex: fx.Array[fx.Int32, LDS_WORDS["tex"], 16]
        tmask: fx.Array[fx.Int32, LDS_WORDS["tmask"], 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(arg_p: fx.Int64, arg_bias: fx.Int64, arg_w: fx.Int64, arg_ids: fx.Int64, i32_m: fx.Int32,
               i32_os: fx.Int32, i32_scale: fx.Int32):
        lane, tok = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_p, num_records_bytes=fx.Int64(ks) * fx.Int64(i32_m) * (E * 4))
        br = buffer_ops.create_buffer_resource_from_addr(arg_bias, num_records_bytes=E * 4)
        out_bytes = fx.Int64(i32_m) * fx.Int64(i32_os) * 4
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=out_bytes)
        ir = buffer_ops.create_buffer_resource_from_addr(arg_ids, num_records_bytes=out_bytes)
        st = topk_new(lane)
        for q in range_constexpr(NQ):
            for sg, k in topk_q_scores(pr, br, lane, tok, i32_m, ks, 0, q):
                st.sig.append(sg)
                st.key.append(k)
        topk_pivot(st, smem, 0, lane)
        if st.nc <= MAXC:
            topk_fast_write(st, smem, 0, lane)
            topk_rank_start(st, smem, 0, lane)
            rank = fx.Int32(0)
            e = fx.Int32(0)
            while e < st.nc:
                rank = topk_rank_step(rank, e, st, smem, 0)
                e = e + RANK_UNROLL
            topk_finish(rank, lane < st.nc, st.mysig, st.myid, lane, smem, 0, wr, ir, tok, i32_os, i32_scale)
        else:
            ss = topk_slow_select(st)
            if ss.total_ties != ss.need:
                topk_tie_masks(ss, smem, 0, lane)
            rank, mysig, myid = topk_slow_out(ss, smem, 0, lane)
            topk_finish(rank, lane < TOPK, mysig, myid, lane, smem, 0, wr, ir, tok, i32_os, i32_scale)

    @flyc.jit
    def launch(arg_p: fx.Int64, arg_bias: fx.Int64, arg_w: fx.Int64, arg_ids: fx.Int64, i32_m: fx.Int32,
               i32_os: fx.Int32, i32_scale: fx.Int32, stream: fx.Stream):
        kernel(arg_p, arg_bias, arg_w, arg_ids, i32_m, i32_os, i32_scale).launch(
            grid=(i32_m, 1, 1), block=(64, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
