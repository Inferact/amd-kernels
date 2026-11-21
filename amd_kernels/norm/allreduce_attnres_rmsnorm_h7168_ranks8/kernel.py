# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank bf16 all-reduce + AttnRes mix + RMSNorm, one workgroup per row.

Workgroup ``row`` of every rank forms a group of eight. Rank ``me`` reduces columns [me * 896, (me + 1) * 896) of the
row over the eight partials (fixed order 0..7, fp32, one bf16 rounding); every rank then adds the prefix, mixes the bank
rows and normalizes the full row itself. Seven compute waves hold two 16 B packs of columns per thread; an eighth wave
reduces this rank's slice, so the bank work and the packets overlap.

Exchange by posted writes into mailboxes, with no flags and no registered inputs: phase 1 writes each peer's slice of
this rank's partial into the peer's mailbox (every compute thread stores its two packs), phase 2 writes the reduced slice
into every peer's mailbox (the eighth wave). A mailbox dword that holds 0xFFFFFFFF is empty (a sent dword of that value,
two NaNs, is sent as 0xFFFFFFFE); a reader polls with system-coherent loads until none of the four dwords of its 16 B
packet is empty, then writes the empty value back. The eighth wave first polls seven probe packets per peer (one load per
lane) and reads every packet once they have arrived; the compute threads poll their two phase-2 packets. Two mailbox
sets per phase alternate with the call parity (a per-row counter on device), so a sender never writes a mailbox its
reader may still have to reset: a rank starts call n + 1 only after it received every peer's phase-2 packets of call n,
which they write after their last phase-1 read of call n. The mailboxes are uncached memory: polled lines are never stale.

The bank rows' squared norms and dot products need no all-reduce result, so they, their softmax weights relative to
their own maximum and their weighted sum ``mixB`` are computed while the packets travel; after the all-reduce only the
prefix source is added: mix = (alpha * mixB + sp * u) / (alpha * denB + sp), and the squared norm of the mix follows
from three block sums (|u|^2, u . w1, u . mixB) and |mixB|^2.
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import const_expr, range_constexpr
from flydsl._mlir.dialects import arith as _arith
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.typing import Vector as Vec

from .norm_math import (
    H, PRELOAD_HINTS, SYS, WORLD, compiler_barrier, exp2, lds_arrive, lds_load16, lds_store16, lds_wait, load_ptrs,
    pk_bf16, rsq, sel8, vstore, wave_sum_lane63,
)

NC = H // 16  # compute threads: thread t owns the 16 B packs of columns 8t and 8(t + NC)
NW = NC // 64
PACKS = 2
POS = H // 8  # 16 B packs of a row
SLICE_T = POS // WORLD  # packs of one rank's slice
SLICE_B = H * 2 // WORLD
ROW_B = H * 2
MAX_ROWS = 256
BANK_ROWS = 8
LOG2E = 1.4426950408889634
MAILBOX_SET = MAX_ROWS * WORLD * SLICE_B  # one phase, one parity: [row][source rank][slice]
BUF_BYTES = 4 * MAILBOX_SET
EMPTY = -1
PROBES = 7  # probe packets per peer slice
PACKED_TILES = 224  # 16 B pack columns of a row / 4: the tile-packed output layout of the MFMA GEMMs

PART_PRE = 0  # 16 values x 8 wave slots: bank squared norms, bank dots
PART_POST = 16  # |u|^2, u . w1, u . mixB, |mixB|^2 (value indices)
PART_N = 20 * 8


def _unpack(pk):
    """(even, odd) columns of a 16 B pack of bf16 as two fp32 vectors of four."""
    return (pk << 16).bitcast(fx.Float32), (pk & -65536).bitcast(fx.Float32)


def _hsum(v):
    return (v[0] + v[1]) + (v[2] + v[3])


def _dot(a, b, acc):
    """acc + a . b over two (even, odd) fp32 vector pairs, kept as a vector of four partial sums."""
    return fx.math.fma(a[1], b[1], fx.math.fma(a[0], b[0], acc))


def _wave_sum(l_part, value):
    """Sum of the seven wave slots of ``value`` in a fixed order (two 16 B LDS reads)."""
    a = lds_load16(l_part + value * 32).bitcast(fx.Float32)
    b = lds_load16(l_part + value * 32 + 16).bitcast(fx.Float32)
    return ((a[0] + a[1]) + (a[2] + a[3])) + ((b[0] + b[1]) + b[2])


def _empty(v):
    return (fx.Int32(v[0]) == EMPTY) | (fx.Int32(v[1]) == EMPTY) | (fx.Int32(v[2]) == EMPTY) | (
        fx.Int32(v[3]) == EMPTY)


def _any_empty(big, n):
    """True if any dword of the first ``n`` packets of ``big`` is the empty marker: the unsigned maximum of the dwords is
    0xFFFFFFFF exactly then (a sent dword never has that value; the loads of an excluded lane return zeros). One
    ``v_max3_u32`` per two dwords instead of a compare chain per packet."""
    m = as_ir_value(fx.Int32(big[0]))
    for k in range(1, 4 * n):
        m = _arith.maxui(m, as_ir_value(fx.Int32(big[k])))
    return fx.Int32(m) == EMPTY


def _sendable(v):
    """The pack with a 0xFFFFFFFF dword changed (such a dword is two bf16 NaNs)."""
    return Vec.from_elements(
        [(fx.Int32(v[i]) == EMPTY).select(fx.Int32(EMPTY - 1), fx.Int32(v[i])) for i in range_constexpr(4)], fx.Int32)


@functools.cache
@launcher
def compile_allreduce_attnres_rmsnorm(has_prefix: bool, out_norm: bool, has_bank: bool, reduce: bool,
                                      packed_out: bool):
    """``reduce``: all-reduce the partials of the eight ranks; otherwise ``arg_x`` is the address of an already
    reduced bf16 [rows, 7168] delta and no rank communicates. ``has_bank``: mix the bank rows; otherwise the output is
    the prefix (an add + RMSNorm). ``packed_out``: the output rows are stored in the (16 rows x 32 columns) tile-packed
    layout of the MFMA GEMMs, rows padded to a multiple of 16."""
    name = ("allreduce_attnres_rmsnorm_h7168_ranks8" + ("_prefix" if has_prefix else "") + ("_norm" if out_norm else "")
            + ("" if has_bank else "_nobank") + ("" if reduce else "_local") + ("_packed" if packed_out else ""))
    block = NC + 64 * reduce

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, PART_N, 16]
        cnt: fx.Array[fx.Int32, 4]
        own: fx.Array[fx.Int32, SLICE_T * 4 if reduce else 4, 16]

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def kernel(
        i32_me: fx.Int32,
        arg_x: fx.Int64,
        arg_bufs: fx.Int64,
        arg_cnt: fx.Int64,
        arg_pre: fx.Int64,
        i32_spre: fx.Int32,
        arg_pout: fx.Int64,
        i32_spout: fx.Int32,
        arg_bank: fx.Int64,
        i32_sbm: fx.Int32,
        i32_sbr: fx.Int32,
        i32_nb: fx.Int32,
        i32_widx: fx.Int32,
        arg_nw: fx.Int64,
        arg_qk: fx.Int64,
        arg_ow: fx.Int64,
        arg_out: fx.Int64,
        i32_eps: fx.Int32,
        i32_oeps: fx.Int32,
    ):
        tx, row = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        l_cnt = fx.Int32(fx.ptrtoint(smem.cnt.ptr))
        l_part = fx.Int32(fx.ptrtoint(smem.part.ptr))
        l_own = fx.Int32(fx.ptrtoint(smem.own.ptr))
        inv_h = fx.Float32(1.0 / H)
        eps = i32_eps.bitcast(fx.Float32)

        def pack(rsrc, o, mask=None, cm=0):
            return Vec(buffer_ops.buffer_load(rsrc, o, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=cm))

        def at(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)

        if const_expr(reduce):
            bs = load_ptrs(arg_bufs)
            cr = at(arg_cnt, MAX_ROWS * 4)
            e = fx.Int32(buffer_ops.buffer_load(cr, row, vec_width=1, dtype=fx.Int32, is_scalar=True))
            par = e & 1
            mine = sel8(i32_me, bs)

            def slot32(phase, src):
                """Byte offset of the (row, src) packet slice in a mailbox set of ``phase`` (0, 1)."""
                return (((phase * 2 + par) * MAX_ROWS + row) * WORLD + src) * SLICE_B

            def slot(phase, src):
                return fx.Int64(slot32(phase, src))

            rs_mine = at(mine, BUF_BYTES)

            def mbox_load(off_bytes, ok):
                """16 B at byte offset ``off_bytes`` of this rank's mailboxes (system-coherent); zeros if not ``ok``."""
                return pack(rs_mine, off_bytes // 4, ok, SYS)

            def mbox_store(off_bytes, v, ok):
                buffer_ops.buffer_store(v, rs_mine, off_bytes // 4, mask=ok, cache_modifier=SYS)

            empty = Vec.from_elements([fx.Int32(EMPTY)] * 4, fx.Int32)
        if tx == 0:
            smem.cnt.ptr[0] = fx.Int32(0)
        fx.gpu.barrier()

        def poll_all(addrs, oks, firsts):
            """The mailbox packets at byte offsets ``addrs`` (those with ``oks`` true), re-read together until none holds
            an empty dword."""
            n = len(addrs)
            big = Vec.from_elements([fx.Int32(f[k]) for f in firsts for k in range(4)], fx.Int32)

            def pending(b):
                return _any_empty(b, n)

            while pending(big):
                compiler_barrier()
                news = [mbox_load(a, ok) for a, ok in zip(addrs, oks)]
                big = Vec.from_elements([fx.Int32(v[k]) for v in news for k in range(4)], fx.Int32)
            return [Vec.from_elements([fx.Int32(big[4 * i + k]) for k in range(4)], fx.Int32) for i in range(n)]

        sl = tx // SLICE_T  # compute thread: slice of its pack 0 (pack 1 lies in slice sl + 4)
        idx = tx - sl * SLICE_T

        def exchange():
            """The eighth wave: phase 1 in (probes, then every packet), reduce, phase 2 out, own slice into LDS."""
            cl = tx - NC
            fx.rocdl.s_setprio(3)
            xrow = at(arg_x + fx.Int64(row) * ROW_B, ROW_B)
            own_idx = [cl, cl + 64]
            own_ok = [cl >= 0, cl < SLICE_T - 64]
            own_part = [pack(xrow, (i32_me * SLICE_T + own_idx[j]) * 4, own_ok[j]) for j in range_constexpr(2)]
            la = [[slot32(0, s) + own_idx[j] * 16 for j in range_constexpr(2)] for s in range_constexpr(WORLD)]
            lok = [[(i32_me != s) & own_ok[j] for j in range_constexpr(2)] for s in range_constexpr(WORLD)]
            sp = cl // PROBES
            pr_ok = cl < (WORLD - 1) * PROBES
            pr_rank = sp + (sp >= i32_me).select(fx.Int32(1), fx.Int32(0))
            pr_addr = slot32(0, pr_rank) + ((cl - sp * PROBES + 1) * (SLICE_T // PROBES) - 1) * 16
            probe = mbox_load(pr_addr, pr_ok)
            while pr_ok & _empty(probe):
                compiler_barrier()
                probe = mbox_load(pr_addr, pr_ok)
            got = [[mbox_load(la[s][j], lok[s][j]) for j in range_constexpr(2)] for s in range_constexpr(WORLD)]
            flat = [(s, j) for j in range(2) for s in range(WORLD)]
            polled = poll_all([la[s][j] for s, j in flat], [lok[s][j] for s, j in flat], [got[s][j] for s, j in flat])
            for q in range_constexpr(len(flat)):
                s, j = flat[q]
                got[s][j] = polled[q]
            red = []
            for j in range_constexpr(2):
                srcs = [(i32_me == r_).select(own_part[j], got[r_][j]) for r_ in range(WORLD)]
                lo = (srcs[0] << 16).bitcast(fx.Float32)
                hi = (srcs[0] & -65536).bitcast(fx.Float32)
                for r_ in range_constexpr(1, WORLD):
                    lo = lo + (srcs[r_] << 16).bitcast(fx.Float32)
                    hi = hi + (srcs[r_] & -65536).bitcast(fx.Float32)
                red.append(Vec.from_elements([pk_bf16(lo[i], hi[i]) for i in range_constexpr(4)], fx.Int32))
            for j in range_constexpr(2):
                send = _sendable(red[j])
                for d in range_constexpr(WORLD):
                    if (i32_me != d) & own_ok[j]:
                        vstore(bs[d] + slot(1, i32_me) + fx.Int64(own_idx[j]) * 16, send)
                if own_ok[j]:
                    lds_store16(l_own + own_idx[j] * 16, red[j])
            for q in range_constexpr(len(flat)):
                s, j = flat[q]
                mbox_store(la[s][j], empty, lok[s][j])
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            if tx == NC:
                buffer_ops.buffer_store(e + 1, cr, row)

        def compute():
            """The seven compute waves: send phase 1, bank statistics and mix, receive phase 2, prefix and norm."""
            part = smem.part.ptr
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            off = [(tx + p * NC) * 4 for p in range(PACKS)]  # dword offsets of this thread's 16 B packs
            zero4 = Vec.filled(4, 0.0, fx.Float32)
            xrow = at(arg_x + fx.Int64(row) * ROW_B, ROW_B)
            xp = [pack(xrow, off[p]) for p in range_constexpr(PACKS)]
            if const_expr(reduce):
                for p in range_constexpr(PACKS):
                    k = sl + 4 * p
                    if k != i32_me:
                        vstore(sel8(k, bs) + slot(0, i32_me) + fx.Int64(idx) * 16, _sendable(xp[p]))
            if const_expr(has_bank):
                bank = at(arg_bank + fx.Int64(row) * fx.Int64(i32_sbm) * 2, (BANK_ROWS * i32_sbr + H) * 2)
                vb = [[pack(bank, r * (i32_sbr // 2) + off[p], mask=i32_nb > r) for p in range_constexpr(PACKS)]
                      for r in range_constexpr(BANK_ROWS)]
                wnw, wqk = at(arg_nw, ROW_B), at(arg_qk, ROW_B)
                w1 = []
                for p in range_constexpr(PACKS):
                    a, b = _unpack(pack(wnw, off[p])), _unpack(pack(wqk, off[p]))
                    w1.append((a[0] * b[0], a[1] * b[1]))
            if const_expr(has_prefix):
                prr = at(arg_pre + fx.Int64(row) * fx.Int64(i32_spre) * 2, ROW_B)
                pv = [pack(prr, off[p]) for p in range_constexpr(PACKS)]
            if const_expr(out_norm):
                wow = at(arg_ow, ROW_B)
                w_out = [_unpack(pack(wow, off[p])) for p in range_constexpr(PACKS)]
            if const_expr(has_bank):
                for r in range_constexpr(BANK_ROWS):
                    if i32_nb > r:
                        s4, d4 = zero4, zero4
                        for p in range_constexpr(PACKS):
                            v = _unpack(vb[r][p])
                            s4 = _dot(v, v, s4)
                            d4 = _dot(v, w1[p], d4)
                        s_w, d_w = wave_sum_lane63(_hsum(s4)), wave_sum_lane63(_hsum(d4))
                        if lane == 63:
                            part[(PART_PRE + r) * 8 + wave] = s_w
                            part[(PART_PRE + BANK_ROWS + r) * 8 + wave] = d_w
                if lane == 0:
                    lds_arrive(l_cnt)
                lds_wait(l_cnt, NW)
                lb = []
                for r in range_constexpr(BANK_ROWS):
                    ss = _wave_sum(l_part, PART_PRE + r)
                    dot = _wave_sum(l_part, PART_PRE + BANK_ROWS + r)
                    lb.append((i32_nb > r).select(dot * rsq(ss * inv_h + eps), fx.Float32(-3.0e38)))
                mb = lb[0]
                for r in range_constexpr(1, BANK_ROWS):
                    mb = (lb[r] > mb).select(lb[r], mb)
                eb = [(i32_nb > r).select(exp2((lb[r] - mb) * LOG2E), fx.Float32(0.0))
                      for r in range_constexpr(BANK_ROWS)]
                den_b = eb[0]
                for r in range_constexpr(1, BANK_ROWS):
                    den_b = den_b + eb[r]
                m0e, m0o, m1e, m1o = zero4, zero4, zero4, zero4
                for r in range_constexpr(BANK_ROWS):
                    if i32_nb > r:
                        a0, a1 = _unpack(vb[r][0]), _unpack(vb[r][1])
                        m0e, m0o = m0e + a0[0] * eb[r], m0o + a0[1] * eb[r]
                        m1e, m1o = m1e + a1[0] * eb[r], m1o + a1[1] * eb[r]
                mix_b = [(m0e, m0o), (m1e, m1o)]
                sbb = zero4
                for p in range_constexpr(PACKS):
                    sbb = _dot(mix_b[p], mix_b[p], sbb)
                sbb = wave_sum_lane63(_hsum(sbb))
            if const_expr(reduce):
                ks = [sl + 4 * p for p in range(PACKS)]
                addrs = [slot32(1, k) + idx * 16 for k in ks]
                oks = [k != i32_me for k in ks]
                recv = poll_all(addrs, oks, [mbox_load(addrs[p], oks[p]) for p in range(PACKS)])
                for p in range_constexpr(PACKS):
                    mbox_store(addrs[p], empty, oks[p])
                fx.gpu.barrier()
                own_d = lds_load16(l_own + idx * 16)
                dpk = [Vec.from_elements([(ks[p] == i32_me).select(fx.Int32(own_d[i]), fx.Int32(recv[p][i]))
                                          for i in range_constexpr(4)], fx.Int32) for p in range_constexpr(PACKS)]
            else:
                dpk = xp
            u, upk = [], []
            for p in range_constexpr(PACKS):
                dv = _unpack(dpk[p])
                if const_expr(has_prefix):
                    pr = _unpack(pv[p])
                    dv = (dv[0] + pr[0], dv[1] + pr[1])
                q = Vec.from_elements([pk_bf16(dv[0][i], dv[1][i]) for i in range_constexpr(4)], fx.Int32)
                upk.append(q)
                u.append(_unpack(q))
            pout = at(arg_pout + fx.Int64(row) * fx.Int64(i32_spout) * 2, ROW_B)
            for p in range_constexpr(PACKS):
                buffer_ops.buffer_store(upk[p], pout, off[p])
                if const_expr(has_bank):
                    buffer_ops.buffer_store(upk[p], bank, i32_widx * (i32_sbr // 2) + off[p], mask=i32_widx >= 0)
            if const_expr(has_bank or out_norm):
                ssu, dotu, bu = zero4, zero4, zero4
                for p in range_constexpr(PACKS):
                    ssu = _dot(u[p], u[p], ssu)
                    if const_expr(has_bank):
                        dotu = _dot(u[p], w1[p], dotu)
                        bu = _dot(u[p], mix_b[p], bu)
                ssu = wave_sum_lane63(_hsum(ssu))
                if const_expr(has_bank):
                    dotu, bu = wave_sum_lane63(_hsum(dotu)), wave_sum_lane63(_hsum(bu))
                if lane == 63:
                    part[PART_POST * 8 + wave] = ssu
                    if const_expr(has_bank):
                        part[(PART_POST + 1) * 8 + wave] = dotu
                        part[(PART_POST + 2) * 8 + wave] = bu
                        part[(PART_POST + 3) * 8 + wave] = sbb
                fx.gpu.barrier()
                ssu = _wave_sum(l_part, PART_POST)
                if const_expr(has_bank):
                    dotu, bu, sbb = (_wave_sum(l_part, PART_POST + 1), _wave_sum(l_part, PART_POST + 2),
                                     _wave_sum(l_part, PART_POST + 3))
            if const_expr(has_bank):
                lp = dotu * rsq(ssu * inv_h + eps)
                m = (mb > lp).select(mb, lp)
                alpha = exp2((mb - m) * LOG2E)
                sp = exp2((lp - m) * LOG2E)
                inv = fx.Float32(1.0) / (alpha * den_b + sp)
                ca, cs = alpha * inv, sp * inv
                if const_expr(out_norm):
                    ssm = ca * ca * sbb + fx.Float32(2.0) * ca * cs * bu + cs * cs * ssu
            else:
                ca, cs = fx.Float32(0.0), fx.Float32(1.0)
                if const_expr(out_norm):
                    ssm = ssu
            if const_expr(out_norm):
                rstd = rsq(ssm * inv_h + i32_oeps.bitcast(fx.Float32))
                ca, cs = ca * rstd, cs * rstd
            if const_expr(packed_out):
                orow = at(arg_out, MAX_ROWS * ROW_B)
            else:
                orow = at(arg_out + fx.Int64(row) * ROW_B, ROW_B)
            for p in range_constexpr(PACKS):
                if const_expr(has_bank):
                    lo = mix_b[p][0] * ca + u[p][0] * cs
                    hi = mix_b[p][1] * ca + u[p][1] * cs
                else:
                    lo, hi = u[p][0] * cs, u[p][1] * cs
                if const_expr(out_norm):
                    lo, hi = lo * w_out[p][0], hi * w_out[p][1]
                opk = Vec.from_elements([pk_bf16(lo[i], hi[i]) for i in range_constexpr(4)], fx.Int32)
                if const_expr(packed_out):
                    tile_pack = (((row >> 4) * PACKED_TILES + (tx >> 2) + SLICE_T * p) * 64 + (tx & 3) * 16
                                 + (row & 15))
                    buffer_ops.buffer_store(opk, orow, tile_pack * 4)
                else:
                    buffer_ops.buffer_store(opk, orow, off[p])

        if const_expr(reduce):
            if tx >= NC:
                exchange()
                fx.gpu.barrier()
                if const_expr(has_bank or out_norm):
                    fx.gpu.barrier()
            else:
                compute()
        else:
            compute()

    @flyc.jit
    def launch(
        i32_m: fx.Int32, i32_me: fx.Int32, arg_x: fx.Int64, arg_bufs: fx.Int64, arg_cnt: fx.Int64,
        arg_pre: fx.Int64, i32_spre: fx.Int32, arg_pout: fx.Int64, i32_spout: fx.Int32, arg_bank: fx.Int64,
        i32_sbm: fx.Int32, i32_sbr: fx.Int32, i32_nb: fx.Int32, i32_widx: fx.Int32, arg_nw: fx.Int64,
        arg_qk: fx.Int64, arg_ow: fx.Int64, arg_out: fx.Int64, i32_eps: fx.Int32, i32_oeps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(i32_me, arg_x, arg_bufs, arg_cnt, arg_pre, i32_spre, arg_pout, i32_spout, arg_bank, i32_sbm,
               i32_sbr, i32_nb, i32_widx, arg_nw, arg_qk, arg_ow, arg_out, i32_eps, i32_oeps).launch(
            grid=(fx.Int64(i32_m), 1, 1), block=(block, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
