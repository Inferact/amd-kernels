# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank latent tail of a Kimi-K3.1 MoE layer in one launch: all-reduce of the [rows, 3584] latent partial,
RMSNorm(3584) and the up projection of this rank's 896-column hidden shard, ``out = shared + rmsnorm(sum_r lat_r) @ W^T``.

Roles (512-thread workgroups). Workgroups 0..127 are the all-reduce: workgroup ``rb`` handles rows RB rb .. RB rb + RB - 1
(one wave per row, RB = 2, 4 or 8 by row count: 40 active all-reduce workgroups next to 168 up-projection ones
make the last up-projection workgroups wait for a CU) when ``rb < nrb``, the others only bump their epoch; the other 224 are
the up projection. Both start at once: an up-projection workgroup loads its weights, waits for the reduced latent, and
runs.

All-reduce (push protocol). Rank ``me`` owns columns [448 me, 448 (me + 1)) of every row. Hop 1: a wave pushes slice d of
its row of the partial into rank d's inbox slot ``me`` (write-through remote stores), waits for the acks and raises
F1 (row, me) at d; when F1 of all seven peers shows this call's epoch it adds the eight slices of its slice in rank
order 0..7 in fp32 (its own from registers) and rounds once to bf16 (the latent). It also does the norm's elementwise
part once for everyone: hop 2 pushes ``xn = bf16(latent * w_norm)`` into the ``lat`` buffer of all eight ranks (itself
included) and the row's sum of squares of its slice into ``lat``'s side array ``ssq[row][me]``; after the workgroup's
wave has had its stores acknowledged it raises F2 (row, me) at every rank. An up-projection workgroup leaves its wait
when F2 (row, o) of its rows and every owner shows the epoch, so a finished kernel has the whole reduced, scaled latent
in its own ``lat`` and every owner has finished reading its inbox, which a peer's next call writes. Epochs are private
per workgroup (one word each, read at the start and written at the end by the same workgroup): every workgroup of the
fixed grid runs once per call and every rank makes the same calls, so all epochs equal the call count. No input address
is exchanged: eager and graph calls are the same.

Up projection. A workgroup owns 16 x NSUB output columns (n tile ``nt``; NSUB is 1 up to 64 rows, else 2) and ``MPG`` 16-token
tiles (m group ``g``; groups_of(rows) groups, so 224 workgroups at most) over the whole K; wave w of its 8 takes the 448 K
columns [448 w, 448 (w + 1)). The weights are the MFMA A operand straight from global memory (16 rows x 4 lane groups, 16 B
per lane and 32 K columns per step) and stay in registers; the B operand is ``xn`` from ``lat`` through LDS (a coalesced
896 B load per row, staged in two halves of 7 K steps, then fragment reads), shared by the NSUB n tiles so that the
activation bytes a CU reads per MFMA halve. The eight waves' accumulators meet in LDS, and the epilogue adds the 8 owners'
sums of squares (``1 / rms`` commutes with the GEMM) and ``shared`` and stores bf16 ``out`` (which may alias ``shared``).
No atomic touches an output and nothing leaves the workgroup before the epilogue. The groups of an n tile share an XCD
(workgroup id % 8): the weights come from HBM once per XCD. The grid is 128 all-reduce slots + 224 up-projection slots for
every call (slots beyond the call's tiles only advance their epoch word), which keeps all epoch words equal.

Modes: ``fused`` (above), ``ar`` (only the all-reduce, waiting for its flags) and ``up`` (only the up projection on a
``lat`` filled by an earlier launch).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .norm_math import (
    K, N, NT, PRELOAD_HINTS, SYS, WORLD, bf16x8, flag_store, load_ptrs, pk_bf16, rsq, sel8, spin_load,
    wave_sum_lane63,
)

SLOTS = 16  # expert slots of the routed experts' output
MAX_SLOT_ROWS = 64  # rows the slot-summing variant takes
RB_MIN = 2  # rows per all-reduce workgroup of the smallest calls (one wave per row)
SL = K // WORLD  # 448 columns of a slice
PK = SL // 8  # 56 packs of 8 bf16 per row slice
MAX_ROWS = 256
MAX_RB = MAX_ROWS // RB_MIN  # all-reduce workgroup slots
MAX_MT = MAX_ROWS // 16
NTILES = N // 16  # 56 n tiles
GROUPS = 4  # m groups (default)
WCM = 0  # cache modifier of the weight loads
NSUB = 2  # 16-column n tiles per workgroup (default)
K2_WGS = 224  # up-projection workgroups of every call
WPE = 0  # waves per SIMD hint (0 = none)
NWAVES = 8
KW = K // NWAVES  # K columns per wave
KSTEPS = KW // 32  # 14
HALF = KSTEPS // 2  # K steps staged at a time (two rounds per tile keep the workgroup's LDS at ~66 KB: two workgroups per CU)
HPK = PK // 2  # 28 packs
XROW = HPK * 16 + 16  # LDS bytes of a staged half row slice: 29 packs of 16 B, so 16 rows hit 16 different bank groups
LINE = 128
F1 = 0  # F1 (row, source): MAX_ROWS x WORLD words
F2 = MAX_RB * LINE  # F2 (row, owner): MAX_ROWS x WORLD words
INBOX = 2 * MAX_RB * LINE
INBOX_BYTES = WORLD * MAX_ROWS * SL * 2
BUF_BYTES = INBOX + INBOX_BYTES
LAT_BYTES = MAX_ROWS * K * 2
SSQ_BYTES = MAX_ROWS * WORLD * 4
LAT_ALLOC = LAT_BYTES + SSQ_BYTES  # the ``lat`` buffer: xn rows, then ssq[row][owner] fp32
CNT_WORDS = 1024
UP_EPOCH0 = MAX_RB


def rb_of(m: int) -> int:
    """Rows per all-reduce workgroup of a call of ``m`` rows: fewer rows keep a workgroup's memory pipe short, but the
    active all-reduce and up-projection workgroups must stay within the 256 CUs (one workgroup each)."""
    return 2 if m <= 64 else 4 if m <= 128 else 8


def nsub_of(mt: int, slots: bool = False) -> int:
    """16-column n tiles per workgroup: one for small calls (less weight traffic per CU before the flags arrive), two
    when the activation rows read per CU dominate."""
    return 1 if mt <= (2 if slots else 4) else 2


def groups_of(mt: int) -> int:
    """m groups of a call of ``mt`` 16-token tiles: an even count (the XCD mapping splits 28 n tiles x groups evenly over 8
    XCDs), at most 8 (224 workgroups), so a workgroup reads few activation rows; the weights it needs are 2 x 114 KB."""
    return 4 if mt <= 4 else 6 if mt <= 6 else 8


def rb_of_slots(m: int) -> int:
    """Rows per all-reduce workgroup when the 16 slots are summed in the kernel: a wave reads 16 slice rows per row."""
    return 1


def mpg_of(mt: int, groups: int = GROUPS) -> int:
    """16-token tiles of an m group for a call of ``mt`` tiles."""
    return (mt + groups - 1) // groups


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(byte_addr)))


def mfma(a, b, c):
    """``c += a x b`` (v_mfma_f32_16x16x32_bf16) as inline asm with the destination tied to ``c``: the allocator gives
    the intrinsic form's destination the registers of a dead B or of a shared zero C, which the 8-pass MFMA still
    reads while it writes. The leading nops cover the VALU -> MFMA operand hazard the compiler cannot see in an asm."""
    r = llvm.inline_asm(
        T.vec(4, T.f32), [fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(c)],
        "s_nop 4\n\tv_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=v,v,v,0", has_side_effects=True)
    return Vec(r)


def mfma_wait(v):
    """``v`` after the wait states its MFMA results need before any read (the asm returns it, so readers follow it)."""
    r = llvm.inline_asm(T.vec(4, T.f32), [fx.as_ir_value(v)], "s_nop 7\n\ts_nop 7", "=v,0", has_side_effects=True)
    return Vec(r)


@launcher
def compile_tail(mode: str, mpg: int, rb: int = 2, ts: bool = False, groups: int = GROUPS, nsub: int = NSUB,
                 slots: bool = False):
    """``mpg``: 16-token tiles of an m group (rows <= 64 mpg); ``ts``: each workgroup stores ``s_memrealtime`` (10 ns) stamps
    of its phases into ``arg_ts`` ([workgroup][wave][8] words, a profiling build)."""
    assert mode in ("fused", "ar", "up") and groups % 2 == 0 and groups * (NTILES // nsub) <= K2_WGS and 1 <= mpg <= (MAX_MT + groups - 1) // groups and rb in ((1, 2) if slots else (2, 4, 8))
    do_ar, do_up = mode != "up", mode != "ar"
    nt2 = NTILES // nsub  # workgroup n tiles of 16 * nsub columns
    RB = rb
    UP0 = MAX_RB if mode == "fused" else 0  # first up-projection workgroup
    LDS = NWAVES * 16 * XROW + NWAVES * mpg * nsub * 1024
    name = f"latent_tail_h3584_n896_{mode}_mpg{mpg}_rb{rb}_g{groups}_n{nsub}" + ("_slots" if slots else "") + ("_ts" if ts else "")
    grid = (MAX_RB if do_ar else 0) + (nt2 * (K2_WGS // nt2) if do_up else 0)  # the grid is the same for every call: all epoch words advance together

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
    def kernel(
        i32_me: fx.Int32,
        arg_bufs: fx.Int64,
        arg_cnt: fx.Int64,
        arg_x: fx.Int64,
        i32_sx: fx.Int32,
        i32_m: fx.Int32,
        i32_nrb: fx.Int32,
        arg_nw: fx.Int64,
        arg_w: fx.Int64,
        arg_shared: fx.Int64,
        i32_ssh: fx.Int32,
        arg_out: fx.Int64,
        i32_sout: fx.Int32,
        i32_eps: fx.Int32,
        i32_zero: fx.Int32,
        arg_ts: fx.Int64,
        arg_rs: fx.Int64,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        bs = load_ptrs(arg_bufs)
        lats = load_ptrs(arg_bufs + WORLD * 8)
        mine = sel8(i32_me, bs)
        my_lat = sel8(i32_me, lats)
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=CNT_WORDS * 4)
        lds0 = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(Shared).peek().buf.ptr))

        def at(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)

        def pack(rsrc, o, mask=None, cm=0):
            return Vec(buffer_ops.buffer_load(rsrc, o, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=cm))

        def epoch(slot):
            return fx.Int32(buffer_ops.buffer_load(cr, slot, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1

        def stamp(k, wait=False):
            if const_expr(ts):
                if const_expr(wait):
                    fx.rocdl.s_waitcnt(vmcnt=0)
                t = llvm.inline_asm(T.i64, [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True)
                buffer_ops.buffer_store(fx.Int64(t).to(fx.Int32), at(arg_ts, 1 << 26), (bid * NWAVES + wave) * 16 + k,
                                        mask=lane == 0)

        def send_done(row, e):
            """F2 (row, me) = e at every rank (the first eight lanes): this owner's slice of ``row`` has landed."""
            flag_store(sel8(lane, bs) + F2 + (fx.Int64(row) * WORLD + fx.Int64(i32_me)) * 4, e)

        def ar_role(rb):
            e = epoch(rb)
            row = rb * RB + wave
            p = lane
            wrow = (wave < RB) & (row < i32_m)
            ok = (lane < PK) & wrow
            stamp(0)
            if rb < i32_nrb:
                wpk = bf16x8(pack(at(arg_nw, K * 2), i32_me * (SL // 2) + p * 4, ((lane < PK) & wrow)))
                if const_expr(slots):
                    # wave d of the workgroup sums, for each of its RB rows, the 16 expert-slot rows of slice d (fp32, slot
                    # order, one bf16 rounding) and pushes it to rank d; one flag per (row, source) follows its acks
                    slr = at(arg_x, 0x7FFFFFF0)
                    rsr = at(arg_rs, MAX_ROWS * SLOTS * 4)
                    for j in range_constexpr(RB):
                        rj = rb * RB + j
                        okj = (lane < PK) & (rj < i32_m)
                        ids = [fx.Int32(buffer_ops.buffer_load(rsr, rj * SLOTS + k, vec_width=1, dtype=fx.Int32,
                                                               is_scalar=True)) for k in range_constexpr(SLOTS)]
                        sp = [pack(slr, ids[k] * (K // 2) + wave * (SL // 2) + p * 4, okj) for k in range_constexpr(SLOTS)]
                        q4 = []
                        for i in range_constexpr(4):
                            lo = (fx.Int32(sp[0][i]) << 16).bitcast(fx.Float32)
                            hi = (fx.Int32(sp[0][i]) & -65536).bitcast(fx.Float32)
                            for k in range_constexpr(1, SLOTS):
                                lo = lo + (fx.Int32(sp[k][i]) << 16).bitcast(fx.Float32)
                                hi = hi + (fx.Int32(sp[k][i]) & -65536).bitcast(fx.Float32)
                            q4.append(pk_bf16(lo, hi))
                        buffer_ops.buffer_store(Vec.from_elements(q4, fx.Int32), at(sel8(wave, bs) + INBOX, INBOX_BYTES),
                                                (i32_me * MAX_ROWS + rj) * (SL // 2) + p * 4, mask=okj,
                                                cache_modifier=SYS)
                    stamp(1, True)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    stamp(2)
                    for j in range_constexpr(RB):
                        rj = rb * RB + j
                        if (lane == j) & (rj < i32_m):
                            flag_store(sel8(wave, bs) + F1 + (fx.Int64(rj) * WORLD + fx.Int64(i32_me)) * 4, e)
                    if (lane < WORLD) & wrow:
                        spin_load(mine + F1 + (fx.Int64(row) * WORLD + fx.Int64(lane)) * 4, e)
                else:
                    xr = at(arg_x, i32_m * i32_sx * 2)
                    mine_pk = [pack(xr, row * (i32_sx // 2) + d * (SL // 2) + p * 4, ok) for d in range_constexpr(WORLD)]
                    for d in range_constexpr(WORLD):
                        dest = at(bs[d] + INBOX, INBOX_BYTES)
                        buffer_ops.buffer_store(mine_pk[d], dest, (i32_me * MAX_ROWS + row) * (SL // 2) + p * 4,
                                                mask=ok & (i32_me != d), cache_modifier=SYS)
                    stamp(1, True)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    stamp(2)
                    if (lane < WORLD) & (lane != i32_me) & wrow:
                        flag_store(sel8(lane, bs) + F1 + (fx.Int64(row) * WORLD + fx.Int64(i32_me)) * 4, e)
                        spin_load(mine + F1 + (fx.Int64(row) * WORLD + fx.Int64(lane)) * 4, e)
                stamp(3)
                inbox = at(mine + INBOX, INBOX_BYTES)
                pk = []
                for s in range_constexpr(WORLD):
                    if const_expr(slots):
                        got = pack(inbox, (s * MAX_ROWS + row) * (SL // 2) + p * 4, ok, SYS)
                        pk.append([fx.Int32(got[i]) for i in range_constexpr(4)])
                    else:
                        got = pack(inbox, (s * MAX_ROWS + row) * (SL // 2) + p * 4, ok & (i32_me != s), SYS)
                        pk.append([(i32_me == s).select(fx.Int32(mine_pk[s][i]), fx.Int32(got[i]))
                                   for i in range_constexpr(4)])
                xn_pk = []
                ssq = fx.Float32(0.0)
                for i in range_constexpr(4):
                    lo = (fx.Int32(pk[0][i]) << 16).bitcast(fx.Float32)
                    hi = (fx.Int32(pk[0][i]) & -65536).bitcast(fx.Float32)
                    for s in range_constexpr(1, WORLD):
                        lo = lo + (fx.Int32(pk[s][i]) << 16).bitcast(fx.Float32)
                        hi = hi + (fx.Int32(pk[s][i]) & -65536).bitcast(fx.Float32)
                    q = pk_bf16(lo, hi)  # the latent, rounded once
                    lo = (q << 16).bitcast(fx.Float32)
                    hi = (q & -65536).bitcast(fx.Float32)
                    ssq = ssq + lo * lo + hi * hi
                    xn_pk.append(pk_bf16(lo * wpk[2 * i], hi * wpk[2 * i + 1]))
                stamp(4, True)
                res = Vec.from_elements(xn_pk, fx.Int32)
                ssq = wave_sum_lane63(ssq)
                for d in range_constexpr(WORLD):
                    lat = at(lats[d], LAT_ALLOC)
                    buffer_ops.buffer_store(res, lat, row * (K // 2) + i32_me * (SL // 2) + p * 4, mask=ok,
                                            cache_modifier=SYS)
                    buffer_ops.buffer_store(ssq, lat, (LAT_BYTES // 4) + row * WORLD + i32_me,
                                            mask=(lane == 63) & wrow, cache_modifier=SYS)
                fx.rocdl.s_waitcnt(vmcnt=0)
                stamp(5)
                if (lane < WORLD) & wrow:
                    send_done(row, e)
                    if const_expr(mode == "ar"):
                        spin_load(mine + F2 + (fx.Int64(row) * WORLD + fx.Int64(lane)) * 4, e)
            buffer_ops.buffer_store(e, cr, rb, mask=tx == 0)
            stamp(6, True)

        def up_role(bid2):
            lds = lds0
            g = lane // 16
            l16 = lane % 16
            xcd = bid2 % 8
            r8 = bid2 // 8
            lin = xcd * (nt2 * groups // 8) + r8  # the groups of an n tile share an XCD (the weights' L2)
            nt = lin // groups
            mg = lin % groups
            n0 = nt * (16 * nsub)
            kb = wave * KW + g * 8  # K column of this lane's pack at step 0; step s adds 32 s
            e2 = epoch(UP_EPOCH0 + bid2)
            m0 = mg * mpg * 16

            def body():
                stamp(0)
                wr = at(arg_w, N * K * 2)
                a_op = [[pack(wr, ((n0 + 16 * u + l16) * K + kb + 32 * s) // 2, None, WCM) for s in range_constexpr(KSTEPS)]
                        for u in range_constexpr(nsub)]
                latr = at(my_lat, LAT_BYTES)
                ssr = at(my_lat + LAT_BYTES, SSQ_BYTES)
                shr = at(arg_shared, i32_m * (i32_ssh * 2))
                outr = at(arg_out, i32_m * (i32_sout * 2))
                eps = i32_eps.bitcast(fx.Float32)
                zero = fx.Int32(i32_zero).bitcast(fx.Float32)
                stamp(1, True)
                if const_expr(mode == "fused"):
                    # wave 0 loads this group's F2 (row, owner) flags together and re-reads them until all show this
                    # call's epoch, then the workgroup goes on
                    f2 = at(mine + F2, MAX_ROWS * WORLD * 4)
                    nrows = ((i32_m - m0 < mpg * 16).select(i32_m - m0, fx.Int32(mpg * 16)))

                    def poll_f2():
                        miss = fx.Int32(0)
                        for j in range_constexpr(mpg * 16 * WORLD // 64):
                            idx = lane + 64 * j
                            need = idx < nrows * WORLD
                            v = fx.Int32(buffer_ops.buffer_load(f2, (m0 + idx // WORLD) * WORLD + idx % WORLD,
                                                                vec_width=1, dtype=fx.Int32, mask=need,
                                                                cache_modifier=SYS))
                            miss = miss | (need & ((v - e2) < 0)).select(fx.Int32(1), fx.Int32(0))
                        return miss

                    if wave == 0:
                        miss = poll_f2()
                        while miss != 0:
                            fx.rocdl.s_sleep(1)
                            llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
                            miss = poll_f2()
                    fx.gpu.barrier()
                stamp(2)

                red = lds + NWAVES * 16 * XROW  # the waves' accumulators meet here
                stage = lds + wave * (16 * XROW)  # this wave's x rows, private

                def load_x(tt):
                    """16 token rows of this wave's 896 B K slice, one coalesced 16 B per lane (lanes 0..55)."""
                    out = []
                    for r in range_constexpr(16):
                        rr = m0 + tt * 16 + r
                        rr = (rr < i32_m).select(rr, i32_m - 1)
                        out.append(pack(latr, (rr * K + wave * KW) // 2 + lane * 4, lane < PK))
                    return out

                def put_x(v, h):
                    if (lane >= HPK * h) & (lane < HPK * (h + 1)):
                        for r in range_constexpr(16):
                            llvm.StoreOp(fx.as_ir_value(v[r]), _lds_ptr(stage + r * XROW + (lane - HPK * h) * 16),
                                         alignment=16)

                nxt = load_x(0)
                ep = tx < nsub * mpg * 64
                sub = tx // (mpg * 64)
                trow = m0 + ((tx // 64) % mpg) * 16 + (tx % 16)
                tcol = n0 + sub * 16 + ((tx % 64) // 16) * 4
                sq0 = Vec(buffer_ops.buffer_load(ssr, trow * WORLD, vec_width=4, dtype=fx.Float32, mask=ep,
                                                 cache_modifier=SYS))
                sq1 = Vec(buffer_ops.buffer_load(ssr, trow * WORLD + 4, vec_width=4, dtype=fx.Float32, mask=ep,
                                                 cache_modifier=SYS))
                sqs = [sq0[0], sq0[1], sq0[2], sq0[3], sq1[0], sq1[1], sq1[2], sq1[3]]
                sh = Vec(buffer_ops.buffer_load(shr, trow * (i32_ssh // 2) + tcol // 2, vec_width=2, dtype=fx.Int32,
                                                mask=ep))
                stamp(3, True)
                for tt in range_constexpr(mpg):
                    cur = nxt
                    accs = [[Vec.filled(4, zero, fx.Float32), Vec.filled(4, zero, fx.Float32)] for u in range_constexpr(nsub)]
                    for h in range_constexpr(2):
                        put_x(cur, h)
                        if const_expr(h == 1 and tt + 1 < mpg):
                            nxt = load_x(tt + 1)  # issued once this tile's registers are in LDS
                        frags = [Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(stage + l16 * XROW + 64 * s + 16 * g),
                                               alignment=16)) for s in range_constexpr(HALF)]
                        for s in range_constexpr(HALF):
                            for u in range_constexpr(nsub):
                                accs[u][s % 2] = mfma(a_op[u][HALF * h + s], frags[s], accs[u][s % 2])
                    for u in range_constexpr(nsub):
                        acc = mfma_wait(accs[u][0]) + mfma_wait(accs[u][1])
                        llvm.StoreOp(fx.as_ir_value(acc), _lds_ptr(red + (((wave * mpg + tt) * nsub + u) * 64 + lane) * 16),
                                     alignment=16)
                stamp(4)
                fx.gpu.barrier()
                stamp(5)
                if ep:
                    slot = tx // 64  # (tile, sub) block of the accumulators: tx // (64) = sub * mpg + tt
                    ln = tx % 64
                    tt = slot % mpg
                    tot = Vec(llvm.load(T.vec(4, T.f32), _lds_ptr(red + (((tt) * nsub + sub) * 64 + ln) * 16), alignment=16))
                    for w2 in range_constexpr(1, NWAVES):
                        tot = tot + Vec(llvm.load(T.vec(4, T.f32),
                                                  _lds_ptr(red + ((((w2 * mpg + tt) * nsub) + sub) * 64 + ln) * 16),
                                                  alignment=16))
                    sq = ((sqs[0] + sqs[1]) + (sqs[2] + sqs[3])) + ((sqs[4] + sqs[5]) + (sqs[6] + sqs[7]))
                    inv = rsq(sq * fx.Float32(1.0 / K) + eps)
                    shf = bf16x8(Vec.from_elements([sh[0], sh[1], fx.Int32(0), fx.Int32(0)], fx.Int32))
                    o = [tot[i] * inv + shf[i] for i in range_constexpr(4)]
                    ov = Vec.from_elements([pk_bf16(o[0], o[1]), pk_bf16(o[2], o[3])], fx.Int32)
                    buffer_ops.buffer_store(ov, outr, trow * (i32_sout // 2) + tcol // 2, mask=trow < i32_m)
                stamp(6, True)

            if (bid2 < nt2 * groups) & (m0 < i32_m):
                body()
            buffer_ops.buffer_store(e2, cr, UP_EPOCH0 + bid2, mask=tx == 0)

        if const_expr(do_ar):
            if bid < MAX_RB:
                ar_role(bid)
        if const_expr(do_up):
            if bid >= UP0:
                up_role(bid - UP0)

    @flyc.jit
    def launch(
        i32_m: fx.Int32, i32_nrb: fx.Int32, i32_me: fx.Int32, arg_bufs: fx.Int64, arg_cnt: fx.Int64,
        arg_x: fx.Int64, i32_sx: fx.Int32, arg_nw: fx.Int64, arg_w: fx.Int64, arg_shared: fx.Int64,
        i32_ssh: fx.Int32, arg_out: fx.Int64, i32_sout: fx.Int32, i32_eps: fx.Int32, i32_zero: fx.Int32, arg_ts: fx.Int64,
        arg_rs: fx.Int64, stream: fx.Stream,
    ):
        kernel(i32_me, arg_bufs, arg_cnt, arg_x, i32_sx, i32_m, i32_nrb, arg_nw, arg_w, arg_shared, i32_ssh,
               arg_out, i32_sout, i32_eps, i32_zero, arg_ts, arg_rs).launch(grid=(grid, 1, 1), block=(512, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = {**PRELOAD_HINTS, **({"waves_per_eu": WPE} if WPE else {})}
    return launch
