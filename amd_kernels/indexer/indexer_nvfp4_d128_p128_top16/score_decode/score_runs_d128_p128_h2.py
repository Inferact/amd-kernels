# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-head decode index scorer over runs of 16 blocks of every request, balanced over the workgroups (fp8 MFMA).

Column c of the one 16-column MFMA group is query row c // 2 of index head c % 2 (``columns.columns``). Work: the
requests' runs (16 consecutive blocks), prefix-summed in LDS; workgroup g takes runs g, g + grid, ..., wave w quarter w
(4 blocks) of each, the next page loading tile by tile under the current one, and stores the quarter's 4 scores of a
(head, row) as 16 B. A call of few runs (device-side lengths, the host's bound) splits each run into four 4-block units
instead, wave w its block w. A wave keeps its request's Q columns in registers and 64 pages' table entries in one
VGPR (readlane per page). The page math and maximum order are the one-head scorers' (same bits).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

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
)
from .columns import columns
from .kernel import BLK, MT, NT, PAGE_BYTES, PAGE_CODES, SENTINEL_INIT, SENTINEL_LOCAL, D, _xlane_max4

NW = 4  # waves per workgroup: quarter w of every run
RUN = 16  # blocks per run
QD = RUN // NW  # blocks per quarter
WIN = 64  # pages per block-table window (one VGPR)
MAX_REQS = 64 * NW  # one request per thread in the prefix sum
NR_LOG = MAX_REQS.bit_length() - 1


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
def compile_score_runs(ql: int):
    """The scorer of ``ql`` (4 or 8) rows per request at two index heads."""
    assert ql in (4, 8)
    QL = ql
    QC, row_of, _ = columns(QL)
    name = f"nvfp4_fp8mfma_index_score_runs_d128_p128_r{QL}h2"

    @fx.struct
    class Shared:
        lut: fx.Array[fx.Int32, 512, 16]  # per scale byte: keys of magnitudes 0..3, 4..7
        incl: fx.Array[fx.Int32, MAX_REQS]  # inclusive prefix of the run counts
        seq: fx.Array[fx.Int32, MAX_REQS]
        wsum: fx.Array[fx.Int32, NW]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_q: fx.Int64,  # [num_reqs * QL, 2, 128] e4m3
        arg_kv: fx.Int64,  # NVFP4 index pages [num_blocks, 128, 72] u8
        arg_lut: fx.Int64,  # nvfp4_e4m3_lut [256] i64
        arg_score: fx.Int64,  # [2, num_reqs * QL, S] fp32
        arg_bt: fx.Int64,  # block table [num_reqs, >= max_blocks] i32
        arg_seq: fx.Int64,  # seq_lens [num_reqs] i32
        i64_page_stride: fx.Int64,  # bytes between pages
        i32_s_head: fx.Int32,  # score stride(0), stride(1) (elements)
        i32_s_row: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,  # leading blocks forced in (sentinel 1e30)
        i32_local: fx.Int32,  # trailing blocks forced in (sentinel 1e29)
        i32_nwg: fx.Int32,  # workgroups (the grid)
        i32_unit_max: fx.Int32,  # units of 4 blocks (one page per wave) up to this many, else runs
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = _uni(tx // 64)
        col, kl = lane % 16, lane // 16
        q_row = row_of(col)
        head = col & 1
        incl_l, seq_l = smem.incl.ptr, smem.seq.ptr

        btr = buffer_ops.create_buffer_resource_from_addr(arg_bt, num_records_bytes=i64_bt_bytes)
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_num_reqs) * (QC * D)
        )
        sr = buffer_ops.create_buffer_resource_from_addr(arg_score, num_records_bytes=i64_s_bytes)

        # e4m3 table; per request its runs, prefix-summed
        lut_g = _global_i32_ptr(arg_lut)
        lut_l = smem.lut.ptr
        for i in range_constexpr(2):
            lut_l[tx * 2 + i] = fx.Int32(lut_g[tx * 2 + i])
        r_ok = tx < i32_num_reqs
        seq_t = fx.Int32(_global_i32_ptr(arg_seq)[r_ok.select(tx, fx.Int32(0))])
        seq_t = r_ok.select(seq_t, fx.Int32(0))
        seq_t = (seq_t < 0).select(fx.Int32(0), seq_t)
        nb_t = (seq_t + (BLK - 1)) // BLK
        runs_t = (nb_t + (RUN - 1)) // RUN
        x = _wave_scan(runs_t, lane, 6)
        if lane == 63:
            smem.wsum.ptr[wave] = x
        seq_l[tx] = seq_t
        fx.gpu.barrier()
        for w in range_constexpr(NW - 1):
            x = x + (wave > w).select(fx.Int32(smem.wsum.ptr[w]), fx.Int32(0))
        incl_l[tx] = x
        fx.gpu.barrier()

        total = _uni(incl_l[MAX_REQS - 1])
        unit = total * NW <= i32_unit_max

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
        # token col's group scales 2 kl, 2 kl + 1: half kl % 2 of a dword
        scale_off = col * 2 + kl // 2  # dwords
        # v_perm selectors: scale byte 2 (kl % 2) (+ 1) of the dword, zeros above
        scale_sel = (kl % 2) * 2 + 0x0C0C0C00

        def page_rsrc(page):
            return buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(page) * i64_page_stride, num_records_bytes=PAGE_BYTES
            )

        def load_tile(kr, ok, t):
            """Tile t of a page: this lane's 16 B of codes (token 16 t + col, dims 32 kl ..) and its group scales;
            ``ok`` false: no memory access."""
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
            """The A fragment of a tile (e4m3 keys, even dims then odd dims per 8)."""
            t0 = lut_load(_perm(sc, sc, scale_sel))
            t1 = lut_load(_perm(sc, sc, scale_sel + 1))
            f = []
            for e in range_constexpr(4):
                lo, hi = _dequant8(fx.Int32(Vec(c)[e]), *(t0 if e < 2 else t1))
                f += [lo, hi]
            return Vec.from_elements(f, fx.Int32)

        def q_load(r, ok):
            """This lane's 32 B of Q (column col of request r, dims 32 kl ..), raw."""
            off = ((r * QC + col) * D + kl * 32) // 4
            okc = ok & (col < QC)
            return [
                Vec(buffer_ops.buffer_load(qr, off + hf * 4, vec_width=4, dtype=fx.Int32, mask=okc))
                for hf in range_constexpr(2)
            ]

        def q_frag(a, b):
            """B operand: the 32 B in the key's dim order (even dims, odd dims per 8)."""
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

        def page_pass(cs, ss, qf, krn, okn, masked, lim):
            """The max over the page's tokens of this lane's column; tile t of the next page loads into tile t's freed
            registers (the barriers keep LLVM from hoisting the loads). Only the reduction depends on the causal
            mask."""
            accs = []
            nxt = []
            for t in range_constexpr(MT):
                f = dequant_tile(cs[t], ss[t])
                nxt.append(load_tile(krn, okn, t))
                accs.append(_mfma_fp8_16x16x128(f, qf, zero4))
                fx.rocdl.sched_barrier(0)
            mx = neg_inf
            if masked:
                run = neg_inf
                for t in range_constexpr(MT):
                    run = reduce(accs[t], t, run, True, lim)
                mx = _xlane_max4(run)
            else:
                run = neg_inf
                for t in range_constexpr(MT):
                    run = reduce(accs[t], t, run, False, None)
                mx = _xlane_max4(run)
            return mx, nxt

        def row_value(mx, seq, q_off, b):
            """AITER's sentinels on block b of the row at q_off (a block both init and local is local)."""
            v1 = seq - QL + q_off + 1  # the row's visible tokens
            rb = ((v1 < 0).select(fx.Int32(0), v1) + (BLK - 1)) // BLK
            l0 = rb - i32_local
            local0 = (l0 < 0).select(fx.Int32(0), l0)
            vis = b < rb
            v = (vis & (b < i32_init)).select(sent_init, mx)
            return (vis & (b >= local0)).select(sent_local, v)

        def scan(UNIT):
            """The pages of this workgroup's items: units (``UNIT``, a quarter of a run, wave w its block w) or runs
            (wave w its quarter w)."""
            qdv = 1 if UNIT else QD  # pages per item and wave
            items = total * NW if UNIT else total
            nitems = (pid < items).select((items - pid + i32_nwg - 1) // i32_nwg, 0)
            nslots = nitems * qdv  # pages of this wave
            w0 = fx.Int32(0)
            while w0 < nslots:
                # the window's pages, one per lane: page s is item pid + (s / qdv) * grid's block (run k = item / 4
                # for units) of request rj (binary search in the prefix)
                s = w0 + lane
                it = pid + (s if UNIT else s >> 2) * i32_nwg
                k = (it >> 2) if UNIT else it
                rj = fx.Int32(0)
                for bs in range_constexpr(NR_LOG):
                    step = MAX_REQS >> (bs + 1)
                    rj = (fx.Int32(incl_l[rj + step - 1]) <= k).select(rj + step, rj)
                valid = s < nslots
                rj = valid.select(rj, fx.Int32(0))
                excl = (rj > 0).select(fx.Int32(incl_l[_max(rj - 1, 0)]), fx.Int32(0))
                seqj = fx.Int32(seq_l[rj])
                nblkj = (seqj + (BLK - 1)) // BLK
                in_run = ((it & 3) * NW + wave) if UNIT else (wave * QD + (s & 3))
                bj = (k - excl) * RUN + in_run
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
                qa, qb = q_load(r_0, n > 0)
                # the loop head merges this path with the back edge: nothing pending
                fx.rocdl.s_waitcnt(vmcnt=0)
                qf = q_frag(qa, qb)
                qna, qnb = qa, qb
                sw = n < 0  # false: Q of the next page already loaded
                v0 = neg_inf
                v1 = neg_inf
                v2 = neg_inf
                v3 = neg_inf
                j = fx.Int32(0)
                while j < n:
                    r = fx.Int32(fx.rocdl.readlane(T.i32, rj, j))
                    b = fx.Int32(fx.rocdl.readlane(T.i32, bj, j))
                    seq = fx.Int32(fx.rocdl.readlane(T.i32, seqj, j))
                    if sw:
                        qf = q_frag(qna, qnb)
                    jn = j + 1
                    okj = jn < n
                    jc = okj.select(jn, fx.Int32(0))
                    pgn = fx.Int32(fx.rocdl.readlane(T.i32, pgj, jc))
                    rn = fx.Int32(fx.rocdl.readlane(T.i32, rj, jc))
                    okn = okj & (pgn >= 0)
                    krn = page_rsrc(okn.select(pgn, fx.Int32(0)))
                    sw = okj & (rn != r)
                    qna, qnb = q_load(rn, sw)
                    fx.rocdl.sched_barrier(0)
                    mx, nxt = page_pass(
                        (c0, c1, c2, c3, c4, c5, c6, c7),
                        (s0, s1, s2, s3, s4, s5, s6, s7),
                        qf,
                        krn,
                        okn,
                        b * BLK + (BLK - 1) > seq - QL,
                        seq - QL + q_row - b * BLK - kl * 4,
                    )
                    c0, s0 = nxt[0]
                    c1, s1 = nxt[1]
                    c2, s2 = nxt[2]
                    c3, s3 = nxt[3]
                    c4, s4 = nxt[4]
                    c5, s5 = nxt[5]
                    c6, s6 = nxt[6]
                    c7, s7 = nxt[7]
                    ov = row_value(mx, seq, q_row, b)
                    off = head * i32_s_head + (r * QL + q_row) * i32_s_row + b
                    nb = (seq + (BLK - 1)) // BLK
                    if const_expr(UNIT):
                        buffer_ops.buffer_store(ov, sr, off, mask=(kl == 0) & (col < QC) & (b < nb))
                    else:
                        e = (w0 + j) & 3
                        v0 = (e == 0).select(ov, v0)
                        v1 = (e == 1).select(ov, v1)
                        v2 = (e == 2).select(ov, v2)
                        v3 = (e == 3).select(ov, v3)
                        if e == QD - 1:
                            # the quarter's 4 blocks of this (head, row) as 16 B (positions past the row's blocks:
                            # unspecified)
                            buffer_ops.buffer_store(
                                Vec.from_elements([_bits(v0), _bits(v1), _bits(v2), _bits(v3)], fx.Int32),
                                sr,
                                off - 3,
                                mask=(kl == 0) & (col < QC) & (b - 3 < nb),
                            )
                    j = jn
                w0 = w0 + WIN

        # one code path per split (selects in the page loop cost run calls ~1%)
        if unit:
            scan(True)
        else:
            scan(False)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_lut: fx.Int64,
        arg_score: fx.Int64,
        arg_bt: fx.Int64,
        arg_seq: fx.Int64,
        i64_page_stride: fx.Int64,
        i32_s_head: fx.Int32,
        i32_s_row: fx.Int32,
        i64_s_bytes: fx.Int64,
        i32_bt_stride: fx.Int32,
        i64_bt_bytes: fx.Int64,
        i32_num_reqs: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_nwg: fx.Int32,
        i32_unit_max: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_q,
            arg_kv,
            arg_lut,
            arg_score,
            arg_bt,
            arg_seq,
            i64_page_stride,
            i32_s_head,
            i32_s_row,
            i64_s_bytes,
            i32_bt_stride,
            i64_bt_bytes,
            i32_num_reqs,
            i32_init,
            i32_local,
            i32_nwg,
            i32_unit_max,
        ).launch(grid=(fx.Int64(i32_nwg), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
