# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Decode index scorer of one of two ranks: all four index heads over the rank's blocks b % 2 == me, the index-q
exchange fused (FlyDSL, gfx950).

Columns: MFMA group g (16 columns) column c is head c % 4, row 4 g + c // 4 of the request (QL = 4: one group, 8: two).
Heads 2 me, 2 me + 1 read this rank's index-q, the other two the peer's rows in this rank's receive buffer. Scores
go to a local buffer [4, rows, S] in half order: block 2 j + me at position j.

Index-q: workgroup g sends this rank's rows of requests g, g + grid, ... to the peer's receive buffer, then a flag per
request; a wave issues its first page's loads before it waits for its window's flags. Workgroup 0 ends with the peer's
flag of request 0, so no rank starts the next call's merge pushes before the peer has finished this call's merge.

Work, by the device-side lengths: up to ``pages_runs`` runs (16 of the rank's blocks of a request), each wave takes a
contiguous range of the rank's pages of all requests in order (balanced to one page; a 4-byte store per page and
(head, row)); above, ``score_decode.score_runs_d128_p128_h2``'s runs: workgroup g takes runs g, g + grid, ..., wave w
quarter w of each, a 16-byte store per quarter and (head, row). Request prefixes of both counts live in LDS; the next
page loads tile by tile under the current one, a request's Q only on a request change. The page pass and maximum
order are ``split4.decode_score``'s (the one-head scorers' bits).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..score_decode.kernel import BLK, MT, NT, PAGE_BYTES, PAGE_CODES, SENTINEL_INIT, SENTINEL_LOCAL, D, _xlane_max4
from ..score_prefill.kernel import (
    PERM_EVEN,
    PERM_ODD,
    _bits,
    _dequant8,
    _global_i32_ptr,
    _maxf_nn,
    _mfma_fp8_16x16x128,
    _pack8,
    _perm,
    _xlane_max4_pair,
)
from .kernel import CNT_MERGE, CNT_SCORE, CNT_WORDS, F_IQ, FLAG_BYTES, HEADS, MAX_REQS, SC1, SYS, WORLD, _flag_read

NW = 4  # waves per workgroup: quarter w of every run
RUN = 16  # this rank's blocks per run
QD = RUN // NW  # blocks per quarter
WIN = 64  # pages per block-table window (one VGPR)
NR_LOG = MAX_REQS.bit_length() - 1
MAX_QL = 8


def _uni(x):
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(x)))


def _min(a, b):
    return (a < b).select(a, b)


def _max(a, b):
    return (a > b).select(a, b)


def _wave_scan(x, lane, steps):
    """Inclusive prefix sum through the first 2**steps lanes."""
    for d in range_constexpr(steps):
        y = fx.Int32(fx.rocdl.ds_bpermute(T.i32, (lane - (1 << d)) * 4, x))
        x = x + (lane >= (1 << d)).select(y, fx.Int32(0))
    return x


@functools.cache
@launcher
def compile_score_half(ql: int):
    """The scorer of ``ql`` (4 or 8) rows per request."""
    assert ql in (4, 8)
    QL = ql
    NG = HEADS * QL // 16  # groups of four rows: every column is a real (head, row)
    QB = QL * 2 * D  # bytes of a request's index-q rows on one rank
    name = f"nvfp4_fp8mfma_index_score_half_ranks2_d128_p128_r{QL}h4"

    @fx.struct
    class Shared:
        lut: fx.Array[fx.Int32, 512, 16]  # per scale byte: keys of magnitudes 0..3, 4..7
        incl: fx.Array[fx.Int32, MAX_REQS]  # inclusive prefix of the run counts
        inclp: fx.Array[fx.Int32, MAX_REQS]  # inclusive prefix of the page counts
        seq: fx.Array[fx.Int32, MAX_REQS]
        wsum: fx.Array[fx.Int32, 2 * NW]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_iq: fx.Int64,  # this rank's index-q [num_reqs * QL, 2, 128] e4m3
        arg_qrcv: fx.Int64,  # this rank's receive buffer of the peer's rows [rmax, 2, 128]
        arg_qpeer: fx.Int64,  # the peer's receive buffer
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] i64
        arg_score: fx.Int64,  # [4, num_reqs * QL, S] fp32, half order
        arg_bt: fx.Int64,  # block table [num_reqs, >= max_blocks] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        arg_cnt: fx.Int64,  # device counters (kernel.CNT_*)
        arg_fmine: fx.Int64,  # this rank's flag words
        arg_fpeer: fx.Int64,  # the peer's
        i32_rmax: fx.Int32,
        i64_page_stride: fx.Int64,
        i32_s_head: fx.Int32,  # score stride(0), stride(1) (elements)
        i32_s_row: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_me: fx.Int32,
        i32_nwg: fx.Int32,  # workgroups (the grid)
        i32_pages_runs: fx.Int32,  # balanced page ranges up to this many runs, else runs
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = _uni(tx // 64)
        col, kl = lane % 16, lane // 16
        st_qoff = col // HEADS
        st_head = col % HEADS
        own = (st_head >> 1) == i32_me
        hl = st_head & 1
        incl_l, seq_l = smem.incl.ptr, smem.seq.ptr

        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=CNT_WORDS * 4)
        calls = _uni(buffer_ops.buffer_load(cr, CNT_SCORE, vec_width=1, dtype=fx.Int32, cache_modifier=SC1)) + 1
        if (pid == 0) & (tx == 0):
            buffer_ops.buffer_store(calls, cr, CNT_MERGE, cache_modifier=SYS)
        btr = buffer_ops.create_buffer_resource_from_addr(arg_bt, num_records_bytes=i64_bt_bytes)
        qbytes = fx.Int64(i32_rmax) * (2 * D)
        ir = buffer_ops.create_buffer_resource_from_addr(arg_iq, num_records_bytes=fx.Int64(i32_num_reqs) * QB)
        qr = buffer_ops.create_buffer_resource_from_addr(arg_qrcv, num_records_bytes=qbytes)
        qp = buffer_ops.create_buffer_resource_from_addr(arg_qpeer, num_records_bytes=qbytes)
        sr = buffer_ops.create_buffer_resource_from_addr(arg_score, num_records_bytes=i64_s_bytes)

        # this rank's index-q rows of requests pid, pid + nwg, ... to the peer
        rs = pid
        while (wave == 0) & (rs < i32_num_reqs):
            for c in range_constexpr(QB // 1024):
                v = buffer_ops.buffer_load(ir, (rs * QB + c * 1024 + lane * 16) // 4, vec_width=4, dtype=fx.Int32)
                buffer_ops.buffer_store(v, qp, (rs * QB + c * 1024 + lane * 16) // 4, cache_modifier=SYS)
            rs = rs + i32_nwg

        # e4m3 table; per request its runs of this rank's blocks, prefix-summed
        lut_g = _global_i32_ptr(arg_lut)
        lut_l = smem.lut.ptr
        for i in range_constexpr(2):
            lut_l[tx * 2 + i] = fx.Int32(lut_g[tx * 2 + i])
        r_ok = tx < i32_num_reqs
        seq_t = fx.Int32(_global_i32_ptr(arg_seq)[r_ok.select(tx, fx.Int32(0))])
        seq_t = r_ok.select(seq_t, fx.Int32(0))
        seq_t = (seq_t < 0).select(fx.Int32(0), seq_t)
        nb_t = (seq_t + (BLK - 1)) // BLK
        mine_t = (nb_t > i32_me).select((nb_t - i32_me + 1) >> 1, fx.Int32(0))
        runs_t = (mine_t + (RUN - 1)) // RUN
        x = _wave_scan(runs_t, lane, 6)
        xp = _wave_scan(mine_t, lane, 6)
        if lane == 63:
            smem.wsum.ptr[wave] = x
            smem.wsum.ptr[NW + wave] = xp
        seq_l[tx] = seq_t
        fx.gpu.barrier()
        for w in range_constexpr(NW - 1):
            x = x + (wave > w).select(fx.Int32(smem.wsum.ptr[w]), fx.Int32(0))
            xp = xp + (wave > w).select(fx.Int32(smem.wsum.ptr[NW + w]), fx.Int32(0))
        incl_l[tx] = x
        smem.inclp.ptr[tx] = xp
        fx.gpu.barrier()

        total = _uni(incl_l[MAX_REQS - 1])
        pages_mode = total <= i32_pages_runs

        lut8 = fx.logical_divide(
            fx.make_view(fx.recast_iter(fx.Int32, smem.lut.ptr), fx.make_layout(512, 1)),
            fx.make_layout(2, 1),
        )
        lut_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

        def lut_load(sbyte):
            r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
            fx.copy(lut_atom, fx.slice(lut8, (None, sbyte)), r)
            v = Vec(r.load())
            return fx.Int32(v[0]), fx.Int32(v[1])

        code_off = (col * 64 + kl * 16) // 4  # dwords
        scale_off = col * 2 + kl // 2  # dwords
        scale_sel = (kl % 2) * 2 + 0x0C0C0C00

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t of a page: this lane's 16 B of codes (token 16 t + col, dims 32 kl ..) and its group scales."""
            c = Vec(
                buffer_ops.buffer_load(
                    kr, code_off, vec_width=4, dtype=fx.Int32, mask=ok, soffset_bytes=t * 1024, cache_modifier=NT
                )
            )
            sc = fx.Int32(
                buffer_ops.buffer_load(
                    kr, scale_off, vec_width=1, dtype=fx.Int32, mask=ok, soffset_bytes=PAGE_CODES + t * 128,
                    cache_modifier=NT,
                )
            )
            return c, sc

        def dequant_tile(c, sc):
            t0 = lut_load(_perm(sc, sc, scale_sel))
            t1 = lut_load(_perm(sc, sc, scale_sel + 1))
            f = []
            for e in range_constexpr(4):
                lo, hi = _dequant8(fx.Int32(Vec(c)[e]), *(t0 if e < 2 else t1))
                f += [lo, hi]
            return Vec.from_elements(f, fx.Int32)

        def q_or(xs, ys):
            return [
                Vec.from_elements([fx.Int32(Vec(x)[e]) | fx.Int32(Vec(y)[e]) for e in range_constexpr(4)], fx.Int32)
                for x, y in zip(xs, ys)
            ]

        def q_part(r, ok, peer):
            """Per group this lane's 32 B of Q (row r * QL + 4 g + st_qoff, head st_head, dims 32 kl ..), raw, from
            this rank's rows (``peer`` False) or the peer's; the other heads' columns read zeros (masked loads)."""
            out = []
            src, m = (qr, ok & ~own) if peer else (ir, ok & own)
            for g in range_constexpr(NG):
                off = (((r * QL + 4 * g + st_qoff) * 2 + hl) * D + kl * 32) // 4
                for hf in range_constexpr(2):
                    out.append(Vec(buffer_ops.buffer_load(src, off + hf * 4, vec_width=4, dtype=fx.Int32, mask=m)))
            return out

        def q_load(r, ok):
            return q_or(q_part(r, ok, False), q_part(r, ok, True))

        def q_frag(a, b):
            halves = []
            for v in (a, b):
                d = [fx.Int32(Vec(v)[e]) for e in range_constexpr(4)]
                halves.append(
                    Vec.from_elements(
                        [
                            _perm(d[1], d[0], PERM_EVEN),
                            _perm(d[1], d[0], PERM_ODD),
                            _perm(d[3], d[2], PERM_EVEN),
                            _perm(d[3], d[2], PERM_ODD),
                        ],
                        fx.Int32,
                    )
                )
            return _pack8(halves[0], halves[1])

        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(float("-inf"))
        sent_init = fx.Float32(SENTINEL_INIT)
        sent_local = fx.Float32(SENTINEL_LOCAL)

        def reduce(v, t, run, masked, lim):
            for e in range_constexpr(4):
                x = v[e]
                if const_expr(masked):
                    x = (lim >= t * 16 + e).select(x, neg_inf)
                run = _maxf_nn(run, x)
            return run

        def page_pass(cs, ss, qf0, qf1, krn, okn, masked, lim):
            """Per group the max over the page's tokens of this lane's column; tile t of the next page loads into tile
            t's freed registers. Only the reduction depends on the causal mask."""
            accs = []
            nxt = []
            for t in range_constexpr(MT):
                f = dequant_tile(cs[t], ss[t])
                nxt.append(load_tile(krn, okn, t))
                if const_expr(NG == 2):
                    accs.append([_mfma_fp8_16x16x128(f, qf0, zero4), _mfma_fp8_16x16x128(f, qf1, zero4)])
                else:
                    accs.append([_mfma_fp8_16x16x128(f, qf0, zero4)])
                fx.rocdl.sched_barrier(0)
            mx0 = neg_inf
            mx1 = neg_inf
            if masked:
                run0 = neg_inf
                run1 = neg_inf
                for t in range_constexpr(MT):
                    run0 = reduce(accs[t][0], t, run0, True, lim)
                    if const_expr(NG == 2):
                        run1 = reduce(accs[t][1], t, run1, True, lim + 4)
                if const_expr(NG == 2):
                    mx0, mx1 = _xlane_max4_pair(run0, run1)
                else:
                    mx0 = _xlane_max4(run0)
            else:
                run0 = neg_inf
                run1 = neg_inf
                for t in range_constexpr(MT):
                    run0 = reduce(accs[t][0], t, run0, False, None)
                    if const_expr(NG == 2):
                        run1 = reduce(accs[t][1], t, run1, False, None)
                if const_expr(NG == 2):
                    mx0, mx1 = _xlane_max4_pair(run0, run1)
                else:
                    mx0 = _xlane_max4(run0)
            return mx0, mx1, nxt

        def row_value(mx, seq, q_off, b):
            """AITER's sentinels on block b of the row at q_off (a block both init and local is local)."""
            v1 = seq - QL + q_off + 1
            rb = ((v1 < 0).select(fx.Int32(0), v1) + (BLK - 1)) // BLK
            l0 = rb - i32_local
            local0 = (l0 < 0).select(fx.Int32(0), l0)
            vis = b < rb
            v = (vis & (b < i32_init)).select(sent_init, mx)
            return (vis & (b >= local0)).select(sent_local, v)

        # the index-q flags follow the pushed rows
        fx.rocdl.s_waitcnt(vmcnt=0)
        fpr = buffer_ops.create_buffer_resource_from_addr(arg_fpeer, num_records_bytes=FLAG_BYTES)
        rs = pid
        while (wave == 0) & (rs < i32_num_reqs):
            buffer_ops.buffer_store(calls, fpr, F_IQ + rs, mask=lane == 0, cache_modifier=SYS)
            rs = rs + i32_nwg

        def scan():
            """Runs mode: the pages of this workgroup's runs, wave w its quarter w."""
            items = total
            nitems = (pid < items).select((items - pid + i32_nwg - 1) // i32_nwg, 0)
            nslots = nitems * QD
            w0 = fx.Int32(0)
            while w0 < nslots:
                s = w0 + lane
                k = pid + (s >> 2) * i32_nwg
                rj = fx.Int32(0)
                for bs in range_constexpr(NR_LOG):
                    step = MAX_REQS >> (bs + 1)
                    rj = (fx.Int32(incl_l[rj + step - 1]) <= k).select(rj + step, rj)
                valid = s < nslots
                rj = valid.select(rj, fx.Int32(0))
                excl = (rj > 0).select(fx.Int32(incl_l[_max(rj - 1, 0)]), fx.Int32(0))
                seqj = fx.Int32(seq_l[rj])
                nblkj = (seqj + (BLK - 1)) // BLK
                in_run = wave * QD + (s & 3)
                jj = (k - excl) * RUN + in_run  # half-order position
                bj = jj * WORLD + i32_me
                frow = seqj - (QL - 1)
                rbf = ((frow < 0).select(fx.Int32(0), frow) + (BLK - 1)) // BLK
                ll = nblkj - i32_local
                loc_last = (ll < 0).select(fx.Int32(0), ll)
                # blocks every row sees whole: init / local ones only need their sentinel
                skip = (bj < rbf) & ((bj < i32_init) | (bj >= loc_last))
                need = valid & (bj < nblkj) & ~skip
                pgj = fx.Int32(
                    buffer_ops.buffer_load(
                        btr, need.select(rj * i32_bt_stride + bj, fx.Int32(0)), vec_width=1, dtype=fx.Int32
                    )
                )
                pgj = need.select(pgj, fx.Int32(-1))
                n = _min(nslots - w0, WIN)

                p_0 = fx.Int32(fx.rocdl.readlane(T.i32, pgj, 0))
                r_0 = fx.Int32(fx.rocdl.readlane(T.i32, rj, 0))
                ok0 = p_0 >= 0
                kr0 = page_rsrc(ok0.select(p_0, fx.Int32(0)))
                fx.rocdl.sched_barrier(0)
                c0, s0 = load_tile(kr0, ok0, 0)
                fx.rocdl.sched_barrier(0)
                c1, s1 = load_tile(kr0, ok0, 1)
                fx.rocdl.sched_barrier(0)
                c2, s2 = load_tile(kr0, ok0, 2)
                fx.rocdl.sched_barrier(0)
                c3, s3 = load_tile(kr0, ok0, 3)
                fx.rocdl.sched_barrier(0)
                c4, s4 = load_tile(kr0, ok0, 4)
                fx.rocdl.sched_barrier(0)
                c5, s5 = load_tile(kr0, ok0, 5)
                fx.rocdl.sched_barrier(0)
                c6, s6 = load_tile(kr0, ok0, 6)
                fx.rocdl.sched_barrier(0)
                c7, s7 = load_tile(kr0, ok0, 7)
                fx.rocdl.sched_barrier(0)
                # this rank's Q of the first page, then the window's requests' index-q rows from the peer: lane i
                # polls its page's request
                q_own = q_part(r_0, n > 0, False)
                want = lane < n
                word = arg_fmine + fx.Int64((F_IQ + rj) * 4)
                f = calls
                if want:
                    f = _flag_read(word)
                while fx.Int64(fx.rocdl.ballot(T.i64, want & (f < calls))) != 0:
                    if want & (f < calls):
                        f = _flag_read(word)
                qr0 = q_or(q_own, q_part(r_0, n > 0, True))
                fx.rocdl.s_waitcnt(vmcnt=0)
                qa0, qb0 = qr0[0], qr0[1]
                qa1, qb1 = qr0[-2], qr0[-1]
                qf0 = q_frag(qa0, qb0)
                qf1 = qf0
                if const_expr(NG == 2):
                    qf1 = q_frag(qa1, qb1)
                qna0, qnb0, qna1, qnb1 = qa0, qb0, qa1, qb1
                sw = n < 0  # false: Q of the next page already loaded
                v00 = neg_inf
                v01 = neg_inf
                v02 = neg_inf
                v03 = neg_inf
                v10 = neg_inf
                v11 = neg_inf
                v12 = neg_inf
                v13 = neg_inf
                j = fx.Int32(0)
                while j < n:
                    r = fx.Int32(fx.rocdl.readlane(T.i32, rj, j))
                    b = fx.Int32(fx.rocdl.readlane(T.i32, bj, j))
                    seq = fx.Int32(fx.rocdl.readlane(T.i32, seqj, j))
                    if sw:
                        qf0 = q_frag(qna0, qnb0)
                        if const_expr(NG == 2):
                            qf1 = q_frag(qna1, qnb1)
                    jn = j + 1
                    okj = jn < n
                    jc = okj.select(jn, fx.Int32(0))
                    pgn = fx.Int32(fx.rocdl.readlane(T.i32, pgj, jc))
                    rn = fx.Int32(fx.rocdl.readlane(T.i32, rj, jc))
                    okn = okj & (pgn >= 0)
                    krn = page_rsrc(okn.select(pgn, fx.Int32(0)))
                    sw = okj & (rn != r)
                    if sw:
                        qrn = q_load(rn, sw)
                        qna0, qnb0 = qrn[0], qrn[1]
                        if const_expr(NG == 2):
                            qna1, qnb1 = qrn[2], qrn[3]
                    fx.rocdl.sched_barrier(0)
                    res = page_pass(
                        (c0, c1, c2, c3, c4, c5, c6, c7),
                        (s0, s1, s2, s3, s4, s5, s6, s7),
                        qf0,
                        qf1,
                        krn,
                        okn,
                        b * BLK + (BLK - 1) > seq - QL,
                        seq - QL + st_qoff - b * BLK - kl * 4,
                    )
                    c0, s0 = res[2][0]
                    c1, s1 = res[2][1]
                    c2, s2 = res[2][2]
                    c3, s3 = res[2][3]
                    c4, s4 = res[2][4]
                    c5, s5 = res[2][5]
                    c6, s6 = res[2][6]
                    c7, s7 = res[2][7]
                    ov0 = row_value(res[0], seq, st_qoff, b)
                    ov1 = ov0
                    if const_expr(NG == 2):
                        ov1 = row_value(res[1], seq, st_qoff + 4, b)
                    pos = b >> 1
                    nb = (seq + (BLK - 1)) // BLK
                    offs = [
                        st_head * i32_s_head + (r * QL + 4 * g + st_qoff) * i32_s_row + pos for g in range_constexpr(NG)
                    ]
                    e = (w0 + j) & 3
                    v00 = (e == 0).select(ov0, v00)
                    v01 = (e == 1).select(ov0, v01)
                    v02 = (e == 2).select(ov0, v02)
                    v03 = (e == 3).select(ov0, v03)
                    if const_expr(NG == 2):
                        v10 = (e == 0).select(ov1, v10)
                        v11 = (e == 1).select(ov1, v11)
                        v12 = (e == 2).select(ov1, v12)
                        v13 = (e == 3).select(ov1, v13)
                    if e == QD - 1:
                        # the quarter's 4 positions of a (head, row) as 16 B (past the row's blocks: unspecified)
                        vsets = [[v00, v01, v02, v03], [v10, v11, v12, v13]]
                        for g in range_constexpr(NG):
                            buffer_ops.buffer_store(
                                Vec.from_elements([_bits(vv) for vv in vsets[g]], fx.Int32),
                                sr,
                                offs[g] - 3,
                                mask=(kl == 0) & (b - 3 * WORLD < nb),
                            )
                    j = jn
                w0 = w0 + WIN

        def scan_pages():
            """Pages mode: the rank's pages of all requests in order, wave gw = pid * NW + wave a contiguous range of
            them (balanced to one page)."""
            totp = _uni(smem.inclp.ptr[MAX_REQS - 1])
            nwv = i32_nwg * NW
            gw = pid * NW + wave
            qd = totp // nwv
            rem = totp - qd * nwv
            start = gw * qd + _min(gw, rem)
            cnt = qd + (gw < rem).select(fx.Int32(1), fx.Int32(0))
            w0 = fx.Int32(0)
            while w0 < cnt:
                s = w0 + lane
                gpg = start + s
                rj = fx.Int32(0)
                for bs in range_constexpr(NR_LOG):
                    step = MAX_REQS >> (bs + 1)
                    rj = (fx.Int32(smem.inclp.ptr[rj + step - 1]) <= gpg).select(rj + step, rj)
                valid = s < cnt
                rj = valid.select(rj, fx.Int32(0))
                excl = (rj > 0).select(fx.Int32(smem.inclp.ptr[_max(rj - 1, 0)]), fx.Int32(0))
                seqj = fx.Int32(seq_l[rj])
                nblkj = (seqj + (BLK - 1)) // BLK
                jj = gpg - excl
                bj = jj * WORLD + i32_me
                frow = seqj - (QL - 1)
                rbf = ((frow < 0).select(fx.Int32(0), frow) + (BLK - 1)) // BLK
                ll = nblkj - i32_local
                loc_last = (ll < 0).select(fx.Int32(0), ll)
                skip = (bj < rbf) & ((bj < i32_init) | (bj >= loc_last))
                need = valid & (bj < nblkj) & ~skip
                pgj = fx.Int32(
                    buffer_ops.buffer_load(
                        btr, need.select(rj * i32_bt_stride + bj, fx.Int32(0)), vec_width=1, dtype=fx.Int32
                    )
                )
                pgj = need.select(pgj, fx.Int32(-1))
                n = _min(cnt - w0, WIN)

                p_0 = fx.Int32(fx.rocdl.readlane(T.i32, pgj, 0))
                r_0 = fx.Int32(fx.rocdl.readlane(T.i32, rj, 0))
                ok0 = p_0 >= 0
                kr0 = page_rsrc(ok0.select(p_0, fx.Int32(0)))
                fx.rocdl.sched_barrier(0)
                c0, s0 = load_tile(kr0, ok0, 0)
                fx.rocdl.sched_barrier(0)
                c1, s1 = load_tile(kr0, ok0, 1)
                fx.rocdl.sched_barrier(0)
                c2, s2 = load_tile(kr0, ok0, 2)
                fx.rocdl.sched_barrier(0)
                c3, s3 = load_tile(kr0, ok0, 3)
                fx.rocdl.sched_barrier(0)
                c4, s4 = load_tile(kr0, ok0, 4)
                fx.rocdl.sched_barrier(0)
                c5, s5 = load_tile(kr0, ok0, 5)
                fx.rocdl.sched_barrier(0)
                c6, s6 = load_tile(kr0, ok0, 6)
                fx.rocdl.sched_barrier(0)
                c7, s7 = load_tile(kr0, ok0, 7)
                fx.rocdl.sched_barrier(0)
                q_own = q_part(r_0, n > 0, False)
                want = lane < n
                word = arg_fmine + fx.Int64((F_IQ + rj) * 4)
                f = calls
                if want:
                    f = _flag_read(word)
                while fx.Int64(fx.rocdl.ballot(T.i64, want & (f < calls))) != 0:
                    if want & (f < calls):
                        f = _flag_read(word)
                qr0 = q_or(q_own, q_part(r_0, n > 0, True))
                fx.rocdl.s_waitcnt(vmcnt=0)
                qa0, qb0 = qr0[0], qr0[1]
                qa1, qb1 = qr0[-2], qr0[-1]
                qf0 = q_frag(qa0, qb0)
                qf1 = qf0
                if const_expr(NG == 2):
                    qf1 = q_frag(qa1, qb1)
                qna0, qnb0, qna1, qnb1 = qa0, qb0, qa1, qb1
                sw = n < 0
                j = fx.Int32(0)
                while j < n:
                    r = fx.Int32(fx.rocdl.readlane(T.i32, rj, j))
                    b = fx.Int32(fx.rocdl.readlane(T.i32, bj, j))
                    seq = fx.Int32(fx.rocdl.readlane(T.i32, seqj, j))
                    if sw:
                        qf0 = q_frag(qna0, qnb0)
                        if const_expr(NG == 2):
                            qf1 = q_frag(qna1, qnb1)
                    jn = j + 1
                    okj = jn < n
                    jc = okj.select(jn, fx.Int32(0))
                    pgn = fx.Int32(fx.rocdl.readlane(T.i32, pgj, jc))
                    rn = fx.Int32(fx.rocdl.readlane(T.i32, rj, jc))
                    okn = okj & (pgn >= 0)
                    krn = page_rsrc(okn.select(pgn, fx.Int32(0)))
                    sw = okj & (rn != r)
                    if sw:
                        qrn = q_load(rn, sw)
                        qna0, qnb0 = qrn[0], qrn[1]
                        if const_expr(NG == 2):
                            qna1, qnb1 = qrn[2], qrn[3]
                    fx.rocdl.sched_barrier(0)
                    res = page_pass(
                        (c0, c1, c2, c3, c4, c5, c6, c7),
                        (s0, s1, s2, s3, s4, s5, s6, s7),
                        qf0,
                        qf1,
                        krn,
                        okn,
                        b * BLK + (BLK - 1) > seq - QL,
                        seq - QL + st_qoff - b * BLK - kl * 4,
                    )
                    c0, s0 = res[2][0]
                    c1, s1 = res[2][1]
                    c2, s2 = res[2][2]
                    c3, s3 = res[2][3]
                    c4, s4 = res[2][4]
                    c5, s5 = res[2][5]
                    c6, s6 = res[2][6]
                    c7, s7 = res[2][7]
                    ov0 = row_value(res[0], seq, st_qoff, b)
                    ov1 = ov0
                    if const_expr(NG == 2):
                        ov1 = row_value(res[1], seq, st_qoff + 4, b)
                    ovs = [ov0, ov1]
                    pos = b >> 1
                    for g in range_constexpr(NG):
                        off = st_head * i32_s_head + (r * QL + 4 * g + st_qoff) * i32_s_row + pos
                        buffer_ops.buffer_store(ovs[g], sr, off, mask=kl == 0)
                    j = jn
                w0 = w0 + WIN

        if pages_mode:
            scan_pages()
        else:
            scan()

        if (pid == 0) & (tx == 0):
            word0 = arg_fmine + fx.Int64(F_IQ * 4)
            f0 = _flag_read(word0)
            while f0 < calls:
                f0 = _flag_read(word0)

    @flyc.jit
    def launch(
        arg_iq: fx.Int64,
        arg_qrcv: fx.Int64,
        arg_qpeer: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        arg_cnt: fx.Int64,
        arg_fmine: fx.Int64,
        arg_fpeer: fx.Int64,
        i32_rmax: fx.Int32,
        i64_page_stride: fx.Int64,
        i32_s_head: fx.Int32,
        i32_s_row: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_me: fx.Int32,
        i32_nwg: fx.Int32,
        i32_pages_runs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_iq,
            arg_qrcv,
            arg_qpeer,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_seq,
            arg_cnt,
            arg_fmine,
            arg_fpeer,
            i32_rmax,
            i64_page_stride,
            i32_s_head,
            i32_s_row,
            i64_s_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_init,
            i32_local,
            i32_me,
            i32_nwg,
            i32_pages_runs,
        ).launch(grid=(fx.Int64(i32_nwg), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
