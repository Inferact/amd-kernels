# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-rank MoE tail in one launch: slot combine + shared add, the reduce-scatter, residual add, Gemma RMSNorm and
MXFP8 quant (row-major codes, AITER-shuffled scales). Three roles share the grid (512 threads per workgroup); this
rank's rows are cut into ``parts`` parts of ``ceil(m / parts)`` rows, and every role but the norm owns whole parts
(dealt round-robin), so one workgroup's wait never stalls the others.

Stagers: per owned part, the slot-order FP32 sum + shared of the peer's rows of that part, rounded once to BF16, into
the own pool slot (counter c -> slot c % 2). A staged row is half-major: 16-byte piece v * 384 + t holds columns
16 t + 8 v .. 16 t + 8 v + 7, the norm lane map's halves, so every wave's stores and loads of a row cover whole lines
(uncached accesses in the natural order use half of every line). ``coded`` (the default): instead every wave codes one
virtual wave (512 columns of a row) per step in ``lossless_x3``'s exact 12-bit code, row r of the part at r x 9600 of
the slot, its raw virtual waves at m x 9600 + r x 12288 (inside the bf16 slot's bytes). Then every wave waits for its
system-coherent stores (vmcnt 0), the workgroup meets at a barrier and lane 0 adds 1 to the part's ticket without
waiting for the result.

Pullers: per owned part, lane 0 spins until the peer's ticket of the part counts this launch, the workgroup acquires
and copies the part (system-coherent loads over the link, system-coherent stores) into a local staging block, then the
same vmcnt / barrier / ticket publishes it locally.

Norm workers: rows strided over the workers; per row lane 0 spins on the local ticket of the row's part, then the
pulled peer row from the staging (coded: decoded, a raw virtual wave read from the peer's slot over the link), the
own slot sum + shared of the row computed in place (rounded once, as the peer's stagers round theirs), the rank-order
sum ``(+0 + b0) + b1`` in FP32 and its BF16 rounding, then
``sum2_add_rmsnorm_mxfp8``'s residual add, norm and quant with its lane map (thread tx holds columns 16 tx ..
16 tx + 15, tx < 384), so the output bits equal the two-launch path's.

Tickets: one uncached buffer per rank, mapped by the peer: the stagers' at byte PEER_TICKETS + 4 j, the pullers' at
LOCAL_TICKETS + 4 j; every launch adds exactly 1 to each, counted against ``calls`` (both ranks launch in the same
order; int32 differences, so the counts may wrap). The launch's end ticket is at END: its last workgroup stores c+1
into the device counter and the launch number into ``calls``. A rank rewrites slot c % 2 only in call c+2, after the
peer finished the pull of call c (the peer's call c+1 launch started after it).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.add_rmsnorm_mxfp8 import NW, _block_sum, _set_fp16_ovfl
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import (
    PRELOAD_HINTS, H, bf16x8, pk_bf16, rsqrt_rn, wave_sum_lane63,
)
from .kernel import SYS, _flag_add, _flag_read, _sel2
from .lossless_x3 import CODED, ROWB, VWAVES, dec16, enc_store, load16
from .slots4_stage_pull import END
from .sum2_add_rmsnorm_mxfp8 import _tail_rowq

NT = 64 * NW
PULLERS = 16
STAGERS = 64
NORMERS = 176
PARTS = 256
UNROLL = 4
PEER_TICKETS = 2048  # byte offsets in the flag buffer, 4 bytes per part
LOCAL_TICKETS = 3072
HALF = H // 4  # dwords per half-major half row


def _slot_sum(arg_slots, arg_shared, row, tx, has, shared_row=None):
    """This thread's 16 columns of row ``row``: bf16((((s0 + s1) + s2) + s3) + shared) as 8 packed dwords; the shared
    row ``shared_row`` (default ``row``), none if ``arg_shared`` is None."""
    sr = buffer_ops.create_buffer_resource_from_addr(arg_slots + fx.Int64(row) * (4 * H * 2),
                                                     num_records_bytes=4 * H * 2)
    if const_expr(arg_shared is not None):
        hr = buffer_ops.create_buffer_resource_from_addr(
            arg_shared + fx.Int64(row if shared_row is None else shared_row) * (H * 2), num_records_bytes=H * 2)
    out = []
    for v in range_constexpr(2):
        vals = [bf16x8(Vec(buffer_ops.buffer_load(sr, q * (H // 2) + tx * 8 + v * 4, vec_width=4, dtype=fx.Int32,
                                                  mask=has)))
                for q in range_constexpr(4)]
        acc = [((vals[0][e] + vals[1][e]) + vals[2][e]) + vals[3][e] for e in range_constexpr(8)]
        if const_expr(arg_shared is not None):
            sh = bf16x8(Vec(buffer_ops.buffer_load(hr, tx * 8 + v * 4, vec_width=4, dtype=fx.Int32, mask=has)))
            acc = [acc[e] + sh[e] for e in range_constexpr(8)]
        out += [pk_bf16(acc[2 * e], acc[2 * e + 1]) for e in range_constexpr(4)]
    return out


def _slot_sum8(arg_slots, arg_shared, row, tid, shared_row=None, shared_mask=None):
    """Code lane ``tid``'s 8 columns of row ``row``, the same sum and rounding as ``_slot_sum``, as 4 packed dwords;
    ``shared_mask``: rows without it add +0 (the owner form's peer rows: only a zero's sign differs, which the
    rank-order sum from +0 drops)."""
    sr = buffer_ops.create_buffer_resource_from_addr(arg_slots + fx.Int64(row) * (4 * H * 2),
                                                     num_records_bytes=4 * H * 2)
    vals = [bf16x8(Vec(buffer_ops.buffer_load(sr, q * (H // 2) + tid * 4, vec_width=4, dtype=fx.Int32)))
            for q in range_constexpr(4)]
    acc = [((vals[0][e] + vals[1][e]) + vals[2][e]) + vals[3][e] for e in range_constexpr(8)]
    if const_expr(arg_shared is not None):
        hr = buffer_ops.create_buffer_resource_from_addr(
            arg_shared + fx.Int64(row if shared_row is None else shared_row) * (H * 2), num_records_bytes=H * 2)
        sh = bf16x8(Vec(buffer_ops.buffer_load(hr, tid * 4, vec_width=4, dtype=fx.Int32, mask=shared_mask)))
        acc = [acc[e] + sh[e] for e in range_constexpr(8)]
    return Vec.from_elements([pk_bf16(acc[2 * e], acc[2 * e + 1]) for e in range_constexpr(4)], fx.Int32)


def compile_slots4_stage_pull_sum2_add_rmsnorm_mxfp8(pullers: int = PULLERS, stagers: int = STAGERS,
                                                     normers: int = NORMERS, parts: int = PARTS, unroll: int = UNROLL,
                                                     coded: bool = True, owner: bool = False):
    """``owner``: ``shared`` is [m, H], this rank's shared expert output of its own rows, added to its own rows' slot
    sums only (the peer's rows carry none); else [2 m, H], added to every row."""
    assert min(pullers, stagers, normers) > 0 and pullers + stagers + normers <= 256
    assert 4 * parts <= LOCAL_TICKETS - PEER_TICKETS
    nwg = pullers + stagers + normers
    step = NT * 4 * unroll
    row_dw = CODED // 4 if coded else H // 2  # dwords per staged row
    name = (f"slots4_stage_pull_sum2_add_rmsnorm_mxfp8_h6144_ranks2_p{pullers}_s{stagers}_n{normers}_c{parts}"
            f"_u{unroll}{'_x3' if coded else ''}{'_owner' if owner else ''}")

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_slots: fx.Int64,  # [rows, 4, H] bf16, every token
        arg_shared: fx.Int64,  # [rows, H] bf16 ([m, H]: owner)
        arg_stage: fx.Int64,  # the peer's sums of this rank's rows: [m, CODED] coded, else [m, H] bf16 half-major
        arg_p0: fx.Int64,
        arg_p1: fx.Int64,
        arg_c0: fx.Int64,
        arg_c1: fx.Int64,
        counter: fx.Tensor,
        calls: fx.Tensor,  # this kernel's launches so far (its monotonic tickets count them)
        i64_stride: fx.Int64,
        i32_rows: fx.Int32,  # every token's rows (2 m)
        i32_me: fx.Int32,
        arg_r: fx.Int64,  # residual [m, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_res: fx.Int64,  # residual out [m, H] bf16
        arg_q: fx.Int64,  # codes e4m3 [m, H] row-major
        arg_s: fx.Int64,  # scales u8, AITER layout
        i64_s_bytes: fx.Int64,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        has = tx < 384
        i32_seq = fx.Int32(counter[0]) + 1
        slot_off = fx.Int64((i32_seq - 1) & 1) * i64_stride
        m = i32_rows // 2
        peer = 1 - i32_me
        my_c = _sel2(i32_me, arg_c0, arg_c1)
        nb = H * 2
        call = fx.Int32(calls[0]) + 1
        c = (m + (parts - 1)) // parts  # rows per part: part j holds rows [min(j c, m), min(j c + c, m))
        if bid < pullers:
            fa = _sel2(peer, arg_c0, arg_c1)
            block = fx.Int64(m) * (row_dw * 4)
            src = _sel2(peer, arg_p0, arg_p1) + slot_off
            if const_expr(not coded):
                src = src + fx.Int64(i32_me) * block
            srs = buffer_ops.create_buffer_resource_from_addr(src, num_records_bytes=block)
            drs = buffer_ops.create_buffer_resource_from_addr(arg_stage, num_records_bytes=block)
            j = bid
            while j < parts:
                if tx == 0:
                    fw = fa + PEER_TICKETS + fx.Int64(j * 4)
                    f = _flag_read(fw)
                    while f - call < 0:
                        f = _flag_read(fw)
                fx.gpu.barrier()
                _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
                r1 = (j * c + c < m).select(j * c + c, m)
                d1 = r1 * row_dw
                off = (j * c < m).select(j * c, m) * row_dw + tx * 4
                while off < d1:
                    vals = []
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        vals.append(Vec(buffer_ops.buffer_load(srs, o, vec_width=4, dtype=fx.Int32, mask=o < d1,
                                                               cache_modifier=SYS)))
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        buffer_ops.buffer_store(vals[u], drs, o, mask=o < d1, cache_modifier=SYS)
                    off = off + step
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx == 0:
                    _flag_add(my_c + LOCAL_TICKETS + fx.Int64(j * 4), 1)
                j = j + pullers
        if (bid >= pullers) & (bid < pullers + stagers):
            sid = bid - pullers
            own_pool = _sel2(i32_me, arg_p0, arg_p1) + slot_off
            dst = buffer_ops.create_buffer_resource_from_addr(own_pool, num_records_bytes=fx.Int64(i32_rows) * nb)
            k = sid
            while k < parts:
                if const_expr(coded):
                    slane = tx % 64
                    it = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
                    r0 = (k * c < m).select(k * c, m)
                    n_it = ((k * c + c < m).select(k * c + c, m) - r0) * VWAVES
                    while it < n_it:
                        row = r0 + it // VWAVES
                        tid = (it % VWAVES) * 64 + slane
                        enc_store(_slot_sum8(arg_slots, None if owner else arg_shared, peer * m + row, tid),
                                  own_pool + fx.Int64(row) * CODED,
                                  own_pool + fx.Int64(m) * CODED + fx.Int64(row) * ROWB, tid)
                        it = it + NW
                else:
                    row = peer * m + (k * c < m).select(k * c, m)
                    r1 = peer * m + (k * c + c < m).select(k * c + c, m)
                    while row < r1:
                        packed = _slot_sum(arg_slots, None if owner else arg_shared, row, tx, has)
                        for v in range_constexpr(2):
                            buffer_ops.buffer_store(Vec.from_elements(packed[v * 4:v * 4 + 4], fx.Int32), dst,
                                                    row * (H // 2) + v * HALF + tx * 4, mask=has,
                                                    cache_modifier=SYS)
                        row = row + 1
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx == 0:
                    _flag_add(my_c + PEER_TICKETS + fx.Int64(k * 4), 1)
                k = k + stagers
        if bid >= pullers + stagers:
            _set_fp16_ovfl()
            nid = bid - pullers - stagers
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=nb)
            wv = []
            for v in range_constexpr(2):
                wv += bf16x8(Vec(buffer_ops.buffer_load(wr, tx * 8 + v * 4, vec_width=4, dtype=fx.Int32, mask=has)))
            qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=fx.Int64(m) * H)
            sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=i64_s_bytes)
            row = nid
            while row < m:
                if tx == 0:
                    fw = my_c + LOCAL_TICKETS + fx.Int64((row // c) * 4)
                    f = _flag_read(fw)
                    while f - call < 0:
                        f = _flag_read(fw)
                fx.gpu.barrier()
                _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="workgroup")
                if const_expr(coded):
                    recs, hw = load16(arg_stage + fx.Int64(row) * CODED, tx, has)
                else:
                    pr = buffer_ops.create_buffer_resource_from_addr(arg_stage + fx.Int64(row) * nb,
                                                                     num_records_bytes=nb)
                    pv = [Vec(buffer_ops.buffer_load(pr, v * HALF + tx * 4, vec_width=4, dtype=fx.Int32, mask=has,
                                                     cache_modifier=SYS)) for v in range_constexpr(2)]
                own = _slot_sum(arg_slots, arg_shared, i32_me * m + row, tx, has, shared_row=row if owner else None)
                if const_expr(coded):
                    pv = dec16(recs, hw, _sel2(peer, arg_p0, arg_p1) + slot_off + fx.Int64(m) * CODED
                               + fx.Int64(row) * ROWB, tx, has)
                rr = buffer_ops.create_buffer_resource_from_addr(arg_r + fx.Int64(row) * nb, num_records_bytes=nb)
                rv = []
                for v in range_constexpr(2):
                    rv += bf16x8(Vec(buffer_ops.buffer_load(rr, tx * 8 + v * 4, vec_width=4, dtype=fx.Int32,
                                                            mask=has)))
                first = i32_me == 0
                b0 = []
                b1 = []
                for v in range_constexpr(2):
                    for i in range_constexpr(4):
                        o = fx.Int32(own[v * 4 + i])
                        p = fx.Int32(pv[v][i])
                        b0.append(first.select(o, p))
                        b1.append(first.select(p, o))
                x0 = bf16x8(b0[:4]) + bf16x8(b0[4:])
                x1 = bf16x8(b1[:4]) + bf16x8(b1[4:])
                acc = [(fx.Float32(0.0) + x0[i]) + x1[i] for i in range_constexpr(16)]
                red = [pk_bf16(acc[2 * i], acc[2 * i + 1]) for i in range_constexpr(8)]
                xv = bf16x8(red[:4]) + bf16x8(red[4:])
                s = [xv[i] + rv[i] for i in range_constexpr(16)]
                res = [pk_bf16(s[2 * i], s[2 * i + 1]) for i in range_constexpr(8)]
                resr = buffer_ops.create_buffer_resource_from_addr(arg_res + fx.Int64(row) * nb,
                                                                   num_records_bytes=nb)
                for v in range_constexpr(2):
                    buffer_ops.buffer_store(Vec.from_elements(res[v * 4:v * 4 + 4], fx.Int32), resr,
                                            tx * 8 + v * 4, mask=has)
                fma = lambda a, b2: fx.Float32(fx.math.fma(a, a, b2 * b2))  # noqa: E731
                e = [fma(s[4 * i], s[4 * i + 2]) for i in range_constexpr(4)]
                o = [fma(s[4 * i + 1], s[4 * i + 3]) for i in range_constexpr(4)]
                ssq = ((e[0] + e[1]) + (e[2] + e[3])) + ((o[0] + o[1]) + (o[2] + o[3]))
                acc2 = wave_sum_lane63(ssq)
                if lane == 63:
                    part[wave] = acc2
                fx.gpu.barrier()
                rstd = rsqrt_rn(_block_sum([part[i] for i in range_constexpr(NW)]), i32_eps.bitcast(fx.Float32))
                _tail_rowq(tx, row, lane, s, wv, rstd, has, qr, sr)
                # every wave has read this row's partial sums before the next row writes them
                fx.gpu.barrier()
                row = row + normers
        fx.gpu.barrier()
        if tx == 0:
            if _flag_add(my_c + END, 1) == nwg - 1:
                _flag_add(my_c + END, -nwg)
                counter[0] = i32_seq
                calls[0] = call

    @flyc.jit
    def launch_slots4_stage_pull_sum2_add_rmsnorm_mxfp8(
        arg_slots: fx.Int64, arg_shared: fx.Int64, arg_stage: fx.Int64, arg_p0: fx.Int64, arg_p1: fx.Int64,
        arg_c0: fx.Int64, arg_c1: fx.Int64, counter: fx.Tensor, calls: fx.Tensor, i64_stride: fx.Int64,
        i32_rows: fx.Int32,
        i32_me: fx.Int32, arg_r: fx.Int64, arg_w: fx.Int64, arg_res: fx.Int64, arg_q: fx.Int64, arg_s: fx.Int64,
        i64_s_bytes: fx.Int64, i32_eps: fx.Int32, stream: fx.Stream,
    ):
        kernel(arg_slots, arg_shared, arg_stage, arg_p0, arg_p1, arg_c0, arg_c1, counter, calls, i64_stride, i32_rows,
               i32_me, arg_r, arg_w, arg_res, arg_q, arg_s, i64_s_bytes, i32_eps).launch(
            grid=(nwg, 1, 1), block=(NT, 1, 1), stream=stream)

    launch_slots4_stage_pull_sum2_add_rmsnorm_mxfp8.kernel_name = name
    launch_slots4_stage_pull_sum2_add_rmsnorm_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_slots4_stage_pull_sum2_add_rmsnorm_mxfp8
