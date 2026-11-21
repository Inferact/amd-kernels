# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""q_b projection, push all-gather of q and W_UK absorption of all 96 heads in one launch, tile by tile.

Grid: ``gp`` producer workgroups, then the consumers (``8 / nw`` per head: 192 of 4 waves or 96 of 8), 256 to 512
threads each at <= 255 VGPRs, so producers and consumers are resident together (``nw`` 4: two workgroups per CU;
``nw`` 8: one, with at most 160 producers). ``stag``: producer ``jj`` of a column block holds its first push back by
``jj stag`` x 0.5 us, so tiles reach the consumers in order.

Producer: a column block of ``16 nt1`` of this rank's 2304 q columns with its weights stationary in registers (K split
over ``nw`` waves) and every ``nsplit``-th 16-token tile of the activation (the next tile's activation loads are in
flight meanwhile). The blocks of one XCD are served by workgroups of that XCD (p % 8), so a block's weights come from
HBM once. A tile's partial sums are reduced in LDS in wave order, rounded once to bf16 and pushed to the gather buffers
of all eight ranks as 512 B x ``nt1`` contiguous runs (the buffer is ordered [source rank, token tile, 16-column
block, token, column]: one run per column block, not a 32 B segment per token), then published as one flag word per
(token tile, column block) once the stores are acknowledged (flags.py).

Consumer: head ``h``, a group of 64 latent columns per wave (``nw`` groups of the head's 8), the W_UK slice in
registers. Rounds: read the flag words of the tiles not yet done until one set is complete, load the q rows of all
tiles that arrived in one batch of system-scope loads (a head's tile is one contiguous 6 KB run), stage them in LDS and
compute ``[bf16(q_nope @ W_UK) | q_pe]`` into out [M, 96, 576] and/or the fp8 hi/lo record of ``fp8q.py`` (out8
[M, 96, 1168]) tile by tile, until every tile is done. ``skeleton``: bit 1 MFMAs replaced by XORs, bit 2 no q_b/x
loads (the producers push what they hold), bit 4 no W_UK loads; ``fp8out`` 3: the consumers only wait (the benchmark's
``gather`` and, with ``skeleton`` 7, ``P0`` columns). ``heads`` 12 and one rank pushing to itself is the single-GPU
loopback of the tests."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from . import fp8q
from .flags import (
    DATA_OFF,
    FREE_OFF,
    LINE,
    Q_AREA,
    READY_OFF,
    WORDS,
    flag_read,
    flag_store,
    sel8,
    spin_from,
    ticket_add,
)
from .kernel_qb import KSTEPS, N, NT, PAD
from .kernel_wuk import GROUPS, KS, L, OW, P, QW, SYS
from .ops import mfma, mfma_stub, now, pk_bf16

NTHR = 256
NW = 4
ROW16 = 25
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def grid_q_push(nt1, nsplit, heads=96, nw=4):
    ncb = N // (16 * nt1)
    return 8 * ((ncb + 7) // 8) * nsplit, (8 // nw) * heads


@launcher
def compile_q_push(nt1, nsplit, mmax, timing=False, skeleton=0, fp8out=0, heads=96, nw=4, stag=0, lay=1):
    """nt1: 16-column tiles of a producer's block; nsplit: producer workgroups per column block (each takes every
    nsplit-th token tile; ``ceil(ncb / 8) nsplit <= 32`` keeps a block's workgroups on one XCD's CUs); mmax: the row bound
    of the variant (a multiple of 16); fp8out: 0 bf16 out, 1 fp8 record only, 2 both; heads: consumer heads (96; 12 only
    in the single-GPU loopback test)."""
    cols = 16 * nt1
    NW = nw
    NTHR = 64 * nw
    assert nw in (4, 8)
    assert fp8out in (0, 1, 2, 3)
    nsl = 8 // nw
    wsteps = KSTEPS // NW
    ncb = N // cols
    ncbh = QW // cols
    assert QW % cols == 0 and mmax % 16 == 0 and ncbh <= 16 and ncb % 1 == 0
    ntb = mmax // 16
    nu = (ntb + nsplit - 1) // nsplit
    gp, _ = grid_q_push(nt1, nsplit, heads, nw)
    cbase = gp
    ncons = nsl * heads
    ncta = max(gp, cbase + ncons)
    assert ntb * ncb <= WORDS and ntb * ncbh <= NTHR and ntb <= 16
    items1 = (16 * (cols // 8) + NTHR - 1) // NTHR
    row4a = (cols + PAD) // 4
    name = (
        f"bf16_q_b_push_w_uk_h96_nt{nt1}_sp{nsplit}_m{mmax}"
        + ("_timing" if timing else "")
        + (f"_skel{skeleton}" if skeleton else "")
        + (f"_fp8{fp8out}" if fp8out else "")
        + (f"_h{heads}" if heads != 96 else "")
        + (f"_nw{nw}" if nw != 4 else "")
        + (f"_st{stag}" if stag else "")
        + ("" if lay else "_rowmajor")
    )
    mac = mfma_stub if skeleton & 1 else mfma
    okb = ntb * 16 * ROW16 * 4
    nrl = (384 + NTHR - 1) // NTHR
    ntw = 0 if fp8out == 3 else ntb
    amx = okb + ntb * ncbh + 8
    full = (1 << ntb) - 1

    @fx.struct
    class Shared:
        m: fx.Array[fx.Int32, max(NW * 16 * (cols + PAD), amx + ntb * NW * 16), 16]

    @flyc.kernel(name=name, known_block_size=[NTHR, 1, 1])
    def kernel(
        x: fx.Int64,
        sx: fx.Int32,
        w_qb: fx.Int64,
        w_uk: fx.Int64,
        out: fx.Int64,
        out8: fx.Int64,
        rows: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        me: fx.Int32,
        state: fx.Int64,
        dbg: fx.Int64,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        bases = (b0, b1, b2, b3, b4, b5, b6, b7)
        mine = sel8(me, bases)
        sr = buffer_ops.create_buffer_resource_from_addr(state, num_records_bytes=16)
        epoch = fx.Int32(buffer_ops.buffer_load(sr, 0, vec_width=1, dtype=fx.Int32, is_scalar=True))
        par = DATA_OFF + fx.Int64(epoch & 1) * Q_AREA
        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        sm = fx.SharedAllocator().allocate(Shared).peek()
        dr = buffer_ops.create_buffer_resource_from_addr(dbg, num_records_bytes=65536 * 64)
        ntb_rt = (rows + 15) // 16
        ip = fx.recast_iter(fx.Int32, sm.m.ptr)

        def stamp(k):
            if timing:
                t_ = now()
                if tx == 0:
                    buffer_ops.buffer_store(
                        Vec.from_elements([t_.to(fx.Int32), (t_ >> fx.Int64(32)).to(fx.Int32)], fx.Int32),
                        dr,
                        bid * 16 + 2 * k,
                    )

        stamp(0)
        xc = bid % 8
        jp = bid // 8
        kk = jp // nsplit
        jj = jp % nsplit
        cb = xc + 8 * kk
        if cb < ncb:
            if (bid == 0) & (tx < 8):
                flag_store(sel8(tx, bases) + FREE_OFF + fx.Int64(me * (LINE * 4)), epoch + 1)
            ffree = flag_read(mine + FREE_OFF + fx.Int64((tx % 8) * (LINE * 4)))
            vp = fx.logical_divide(
                fx.make_view(ip, fx.make_layout(NW * 16 * (cols + PAD), 1)),
                fx.make_layout(4, 1),
            )
            wr = buffer_ops.create_buffer_resource_from_addr(w_qb, num_records_bytes=N * 1536 * 2)
            xr = buffer_ops.create_buffer_resource_from_addr(x, num_records_bytes=fx.Int64(rows) * fx.Int64(sx * 2))
            dsts = [
                buffer_ops.create_buffer_resource_from_addr(
                    bases[p] + par, num_records_bytes=Q_AREA if lay else fx.Int64(rows) * (96 * QW * 2)
                )
                for p in range_constexpr(8)
            ]

            def load_x(tb):
                vs = []
                if const_expr(skeleton & 2):
                    return [Vec.filled(4, 0, fx.Int32) for _ in range_constexpr(wsteps)]
                for s in range_constexpr(wsteps):
                    tok = tb * 16 + lane % 16
                    vs.append(
                        Vec(
                            buffer_ops.buffer_load(
                                xr,
                                tok * (sx // 2) + (wave * wsteps + s) * 16 + (lane // 16) * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=(tok < rows) & (tb < ntb_rt),
                            )
                        )
                    )
                return vs

            aw = []
            for s in range_constexpr(wsteps):
                aw.append(
                    [
                        Vec.filled(4, 0, fx.Int32)
                        if skeleton & 2
                        else Vec(
                            buffer_ops.buffer_load(
                                wr,
                                ((cb * nt1 + t) * KSTEPS + wave * wsteps + s) * 256 + lane * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for t in range_constexpr(nt1)
                    ]
                )
            xcur = load_x(jj)
            stamp(7)
            for u in range_constexpr(nu):
                tb = jj + nsplit * u
                xnext = load_x(tb + nsplit) if u + 1 < nu else None
                if tb < ntb_rt:
                    acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(nt1)]
                    for s in range_constexpr(wsteps):
                        for t in range_constexpr(nt1):
                            acc[t] = mac(aw[s][t], xcur[s], acc[t])
                    for t in range_constexpr(nt1):
                        tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        tmp.store(acc[t].bitcast(fx.Int32))
                        fx.copy(atom, tmp, fx.slice(vp, (None, (wave * 16 + lane % 16) * row4a + t * 4 + lane // 16)))
                    if u == 0:
                        stamp(1)
                        if const_expr(heads == 96) and tx < 8:
                            spin_from(mine + FREE_OFF + fx.Int64(tx * (LINE * 4)), epoch, ffree)
                        if const_expr(stag > 0):
                            sn = fx.Int32(0)
                            while sn < jj * stag:
                                fx.rocdl.s_sleep(19)
                                sn = sn + 1
                    fx.gpu.barrier()
                    for r in range_constexpr(items1):
                        item = tx + r * NTHR
                        if const_expr(lay):
                            ch2 = item // 32
                            tk = (item % 32) // 2
                            ch = ch2 * 2 + item % 2
                        else:
                            tk = item // (cols // 8)
                            ch = item % (cols // 8)
                        tok = tb * 16 + tk
                        live = (tk < 16) & (tok < rows)
                        if const_expr(lay):
                            live = live & (ch2 < nt1)
                        sums = [None] * 8
                        for w in range_constexpr(NW):
                            for h in range_constexpr(2):
                                tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                                fx.copy(atom, fx.slice(vp, (None, (w * 16 + tk) * row4a + ch * 2 + h)), tmp)
                                v = Vec(tmp.load()).bitcast(fx.Float32)
                                for e in range_constexpr(4):
                                    a = fx.Float32(v[e])
                                    sums[h * 4 + e] = a if w == 0 else sums[h * 4 + e] + a
                        packed = Vec.from_elements(
                            [pk_bf16(sums[2 * q], sums[2 * q + 1]) for q in range_constexpr(4)], fx.Int32
                        )
                        if const_expr(lay):
                            doff = (((me * 16 + tb) * 144 + cb * nt1 + ch2) * 16 + tk) * 8 + (item % 2) * 4
                        else:
                            doff = tok * (96 * 96) + me * 1152 + cb * (cols // 2) + ch * 4
                        for p in range_constexpr(8):
                            buffer_ops.buffer_store(packed, dsts[p], doff, mask=live)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    if u == 0:
                        stamp(4)
                    if tx < 8:
                        flag_store(sel8(tx, bases) + READY_OFF + fx.Int64((me * WORDS + tb * ncb + cb) * 4), epoch + 1)
                    stamp(5 if u == 0 else 2)
                xcur = xnext if xnext is not None else xcur
        if (bid >= cbase) & (bid < cbase + ncons):
            sl = (bid - cbase) % nsl
            h = (bid - cbase) // nsl
            v16 = fx.logical_divide(
                fx.make_view(ip, fx.make_layout(ntb * 16 * ROW16 * 4, 1)),
                fx.make_layout(4, 1),
            )
            wr2 = buffer_ops.create_buffer_resource_from_addr(w_uk, num_records_bytes=96 * L * P * 2)
            gq = buffer_ops.create_buffer_resource_from_addr(
                mine + par, num_records_bytes=Q_AREA if lay else fx.Int64(rows) * (96 * QW * 2)
            )
            dst = buffer_ops.create_buffer_resource_from_addr(out, num_records_bytes=fx.Int64(rows) * (96 * OW * 2))
            dst8 = buffer_ops.create_buffer_resource_from_addr(
                out8, num_records_bytes=fx.Int64(rows) * (96 * fp8q.RECORD)
            )
            grp = sl * nw + wave
            aw = []
            for jt in range_constexpr(4):
                for s in range_constexpr(KS):
                    aw.append(
                        Vec.filled(4, 0, fx.Int32)
                        if (skeleton & 4) or fp8out == 3
                        else Vec(
                            buffer_ops.buffer_load(
                                wr2,
                                (((h * GROUPS + grp) * 4 + jt) * KS + s) * 256 + lane * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                cache_modifier=NT,
                            )
                        )
                    )
            stamp(3)
            def compute_tile(ti):
                bq = []
                for s in range_constexpr(KS):
                    tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(
                        atom, fx.slice(v16, (None, (ti * 16 + lane % 16) * ROW16 + s * 4 + lane // 16)), tmp
                    )
                    bq.append(Vec(tmp.load()))
                tok = ti * 16 + lane % 16
                accs = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)]
                for s in range_constexpr(KS):
                    for jt in range_constexpr(4):
                        accs[jt] = mac(aw[jt * KS + s], bq[s], accs[jt])
                packed = []
                for jt in range_constexpr(4):
                    packed += [pk_bf16(accs[jt][2 * z], accs[jt][2 * z + 1]) for z in range_constexpr(2)]
                if const_expr(fp8out != 1):
                    for z in range_constexpr(2):
                        buffer_ops.buffer_store(
                            Vec.from_elements(packed[4 * z : 4 * z + 4], fx.Int32),
                            dst,
                            tok * (96 * OW // 2) + h * (OW // 2) + grp * 32 + z * 16 + (lane // 16) * 4,
                            mask=tok < rows,
                        )
                if const_expr(fp8out != 0):
                    am = fp8q.xmax(fp8q.abs_max_bits(packed))
                    if lane < 16:
                        fx.ptr_store(
                            Vec.from_elements([am.bitcast(fx.Int32)], fx.Int32),
                            ip + (amx + ti * NW * 16 + wave * 16 + lane),
                        )
                    fx.gpu.barrier()
                    pv = fx.ptr_load(
                        ip + (amx + ti * NW * 16 + (wave ^ 1) * 16 + lane % 16),
                        result_type=Vec.make_type(1, fx.Int32),
                    )
                    part = fx.Float32(Vec(pv)[0].bitcast(fx.Float32))
                    s_b, inv = fp8q.e8m0_for(fp8q.fmax(am, part))
                    rec = tok * 96 + h
                    for z in range_constexpr(2):
                        xs = []
                        for d in packed[4 * z : 4 * z + 4]:
                            xs += list(fp8q.bf16_pair(d))
                        hi, lo = fp8q.quant(xs, inv)
                        off_hi = rec * (fp8q.RECORD // 4) + (grp * 64 + z * 32 + (lane // 16) * 8) // 4
                        buffer_ops.buffer_store(Vec.from_elements(hi, fx.Int32), dst8, off_hi, mask=tok < rows)
                        buffer_ops.buffer_store(
                            Vec.from_elements(lo, fx.Int32), dst8, off_hi + fp8q.Q_LO // 4, mask=tok < rows
                        )
                    if (wave % 2 == 0) & (lane < 16) & (tok < rows):
                        fp8q.store_byte(s_b, dst8, rec * fp8q.RECORD + fp8q.Q_SCALE + grp // 2)
            def copy_pe(ti):
                if tx < 128:
                    tk = tx // 8
                    ch = tx % 8
                    gt = ti * 16 + tk
                    live = (gt < rows) & ((gt % nsl) == sl)
                    if live:
                        tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(atom, fx.slice(v16, (None, (ti * 16 + tk) * ROW16 + 16 + ch)), tmp)
                        if const_expr(fp8out != 1):
                            buffer_ops.buffer_store(
                                Vec(tmp.load()), dst, gt * (96 * OW // 2) + h * (OW // 2) + 256 + ch * 4, mask=live
                            )
                        if const_expr(fp8out != 0):
                            buffer_ops.buffer_store(
                                Vec(tmp.load()),
                                dst8,
                                (gt * 96 + h) * (fp8q.RECORD // 4) + fp8q.Q_ROPE // 4 + ch * 4,
                                mask=live,
                            )
            rd_base = mine + READY_OFF + fx.Int64(((h // 12) * WORDS + (h % 12) * ncbh) * 4)
            done = fx.Int32(full) - ((fx.Int32(1) << ntb_rt) - 1)
            first = True

            while done != full:
                todo = fx.Int32(0)
                while todo == 0:
                    if tx < ntb * ncbh:
                        v = flag_read(rd_base + fx.Int64(((tx // ncbh) * ncb + tx % ncbh) * 4))
                        fx.ptr_store(
                            Vec.from_elements([((v - (epoch + 1)) >= 0).select(fx.Int32(1), fx.Int32(0))], fx.Int32),
                            ip + (okb + tx),
                        )
                    fx.gpu.barrier()
                    nt_ = fx.Int32(0)
                    for ti in range_constexpr(ntb):
                        arr = fx.Int32(1)
                        for w in range_constexpr(ncbh):
                            pv = fx.ptr_load(ip + (okb + ti * ncbh + w), result_type=Vec.make_type(1, fx.Int32))
                            arr = arr & Vec(pv)[0]
                        nt_ = nt_ | (((done >> ti) & 1) == 0).select(arr << ti, fx.Int32(0))
                    todo = fx.Int32(fx.rocdl.readfirstlane(T.i32, nt_))
                    fx.gpu.barrier()
                if first:
                    stamp(2)
                    first = False
                stamp(1)
                loaded = []
                for ti in range_constexpr(ntw):
                    on = ((todo >> ti) & 1) != 0
                    for r in range_constexpr(nrl):
                        item = tx + r * NTHR
                        if const_expr(lay):
                            tk = (item % 32) // 2
                            goff = (((h // 12) * 16 + ti) * 144 + (h % 12) * 12) * 128 + item * 4
                        else:
                            tk = item // 24
                            goff = (ti * 16 + tk) * (96 * QW // 2) + h * (QW // 2) + (item % 24) * 4
                        loaded.append(
                            Vec(
                                buffer_ops.buffer_load(
                                    gq,
                                    goff,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    mask=on & (item < 384) & (ti * 16 + tk < rows),
                                    cache_modifier=SYS,
                                )
                            )
                        )
                for ti in range_constexpr(ntw):
                    on = ((todo >> ti) & 1) != 0
                    for r in range_constexpr(nrl):
                        item = tx + r * NTHR
                        if const_expr(lay):
                            tk = (item % 32) // 2
                            c = (item // 32) * 2 + item % 2
                        else:
                            tk = item // 24
                            c = item % 24
                        if on & (item < 384) & (ti * 16 + tk < rows):
                            tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            tmp.store(loaded[ti * nrl + r])
                            fx.copy(atom, tmp, fx.slice(v16, (None, (ti * 16 + tk) * ROW16 + c)))
                fx.gpu.barrier()
                stamp(4)
                for ti in range_constexpr(ntw):
                    if ((todo >> ti) & 1) != 0:
                        compute_tile(ti)
                for ti in range_constexpr(ntw):
                    if ((todo >> ti) & 1) != 0:
                        copy_pe(ti)
                done = done | todo
                fx.gpu.barrier()
                stamp(5)

        stamp(6)
        fx.gpu.barrier()
        if tx == 0:
            old = ticket_add(state + 4)
            if old + 1 == ncta:
                flag_store(state + 4, 0)
                flag_store(state, epoch + 1)

    @flyc.jit
    def launch_push2(
        x: fx.Int64,
        sx: fx.Int32,
        w_qb: fx.Int64,
        w_uk: fx.Int64,
        out: fx.Int64,
        out8: fx.Int64,
        rows: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        me: fx.Int32,
        state: fx.Int64,
        dbg: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            x, sx, w_qb, w_uk, out, out8, rows, b0, b1, b2, b3, b4, b5, b6, b7, me, state, dbg,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": f"{NTHR},{NTHR}"},
        ).launch(grid=(fx.Int64(ncta), 1, 1), block=(NTHR, 1, 1), stream=stream)

    launch_push2.compile_hints = PRELOAD_HINTS
    launch_push2.kernel_name = name
    return launch_push2
