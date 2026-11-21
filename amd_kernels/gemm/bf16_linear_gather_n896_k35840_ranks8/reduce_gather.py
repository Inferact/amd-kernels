# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Slab reduce, eight-rank column gather and RMSNorm of the sharded projection, one workgroup per row.

Workgroup ``row`` of every rank forms a group of eight. It sums the K-split fp32 slabs of its rank's 896 columns (two
halves of the slabs on two thread groups, fixed order), rounds once to bf16 and posts the 1792 B slice into each peer's
mailbox; then it collects the eight slices of the row (its own from registers), normalizes the 7168 columns and writes
the row.

Mailboxes carry no flags and need no registered inputs: a dword that holds 0xFFFFFFFF is empty (a sent dword of that
value, two NaNs, is sent as 0xFFFFFFFE); a reader polls with loads until none of the four dwords of its 16 B packet is
empty, then writes the empty value back. Two mailbox sets alternate with the call parity (a per-row counter on device),
so a sender never writes a mailbox its reader may still have to reset: a rank starts call n + 2 only after it received
every peer's slice of call n + 1, which they post after their call n finished.
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .device import (PRELOAD_HINTS, WORLD, compiler_barrier, lds_load16, lds_store16, load_ptrs, pk_bf16, rsq, sel8,
                     vload, vstore, wave_sum_lane63)
from .reference import N_FULL, N_SHARD

MAX_ROWS = 256
SLICE_B = N_SHARD * 2
THREADS = 256
CT = N_SHARD // 8  # threads of one half: 8 columns each
EMPTY = -1
SET_BYTES = MAX_ROWS * WORLD * SLICE_B
BUF_BYTES = 2 * SET_BYTES
COUNTS = MAX_ROWS


def _empty(v):
    return (fx.Int32(v[0]) == EMPTY) | (fx.Int32(v[1]) == EMPTY) | (fx.Int32(v[2]) == EMPTY) | (
        fx.Int32(v[3]) == EMPTY)


def _sendable(v):
    return Vec.from_elements(
        [(fx.Int32(v[i]) == EMPTY).select(fx.Int32(EMPTY - 1), fx.Int32(v[i])) for i in range_constexpr(4)], fx.Int32)


def _unpack(pk):
    """The eight fp32 of a 16 B pack of bf16."""
    out = []
    for i in range_constexpr(4):
        w = fx.Int32(pk[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


@functools.cache
@launcher
def compile_reduce_gather(split: int):
    name = f"bf16_reduce_gather_norm_n896_ranks8_sk{split}"
    half = (split + 1) // 2

    @fx.struct
    class Shared:
        acc: fx.Array[fx.Int32, CT * 8, 16]
        ss: fx.Array[fx.Float32, 4]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(i32_me: fx.Int32, i32_rows: fx.Int32, arg_slab: fx.Int64, arg_bufs: fx.Int64, arg_cnt: fx.Int64,
               arg_w: fx.Int64, arg_out: fx.Int64, i32_eps: fx.Int32):
        tx, row = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        l_acc = fx.Int32(fx.ptrtoint(smem.acc.ptr))
        grp = (tx >= CT).select(fx.Int32(1), fx.Int32(0))
        c = tx - grp * CT
        live = tx < 2 * CT
        bs = load_ptrs(arg_bufs)
        mine = sel8(i32_me, bs)
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=COUNTS * 4)
        e = fx.Int32(buffer_ops.buffer_load(cr, row, vec_width=1, dtype=fx.Int32, is_scalar=True))
        par = e & 1

        def slot(src):
            return fx.Int64((par * MAX_ROWS + row) * WORLD + src) * SLICE_B

        slab = buffer_ops.create_buffer_resource_from_addr(arg_slab, num_records_bytes=fx.Int64(2 ** 31))
        acc = [fx.Float32(0.0) for _ in range_constexpr(8)]
        for j in range_constexpr(half):
            s = 2 * j + grp
            ok = live & (s < split)
            base = (fx.Int32(s) * i32_rows + row) * N_SHARD + c * 8
            lo = Vec(buffer_ops.buffer_load(slab, base, vec_width=4, dtype=fx.Float32, mask=ok))
            hi = Vec(buffer_ops.buffer_load(slab, base + 4, vec_width=4, dtype=fx.Float32, mask=ok))
            for i in range_constexpr(4):
                acc[i] = acc[i] + ok.select(fx.Float32(lo[i]), fx.Float32(0.0))
                acc[4 + i] = acc[4 + i] + ok.select(fx.Float32(hi[i]), fx.Float32(0.0))
        if live & (grp == 1):
            lds_store16(l_acc + c * 32, Vec.from_elements([a.bitcast(fx.Int32) for a in acc[:4]], fx.Int32))
            lds_store16(l_acc + c * 32 + 16, Vec.from_elements([a.bitcast(fx.Int32) for a in acc[4:]], fx.Int32))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        lo = lds_load16(l_acc + c * 32)
        hi = lds_load16(l_acc + c * 32 + 16)
        tot = [acc[i] + fx.Int32(lo[i]).bitcast(fx.Float32) for i in range_constexpr(4)] + [
            acc[4 + i] + fx.Int32(hi[i]).bitcast(fx.Float32) for i in range_constexpr(4)]
        own = Vec.from_elements([pk_bf16(tot[2 * i], tot[2 * i + 1]) for i in range_constexpr(4)], fx.Int32)
        send = _sendable(own)
        mine_ok = live & (grp == 0)
        for d in range_constexpr(WORLD):
            if mine_ok & (i32_me != d):
                vstore(bs[d] + slot(i32_me) + fx.Int64(c) * 16, send)
        empty = Vec.from_elements([fx.Int32(EMPTY)] * 4, fx.Int32)
        la = [(mine_ok & (i32_me != s)).select(mine + slot(s) + fx.Int64(c) * 16, mine) for s in range_constexpr(WORLD)]
        got = [vload(la[s]) for s in range_constexpr(WORLD)]
        wres = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N_FULL * 2)
        wts = [_unpack(Vec(buffer_ops.buffer_load(wres, (s * N_SHARD + c * 8) // 2, vec_width=4, dtype=fx.Int32,
                                                  mask=mine_ok))) for s in range_constexpr(WORLD)]
        ssq = fx.Float32(0.0)
        vals = []
        for s in range_constexpr(WORLD):
            v = got[s]
            while (mine_ok & (i32_me != s)) & _empty(v):
                compiler_barrier()
                v = vload(la[s])
            if mine_ok & (i32_me != s):
                vstore(la[s], empty)
            words = [(i32_me == s).select(fx.Int32(own[i]), fx.Int32(v[i])) for i in range_constexpr(4)]
            f = _unpack(words)
            vals.append(f)
            for i in range_constexpr(8):
                ssq = ssq + (mine_ok.select(f[i], fx.Float32(0.0))) * (mine_ok.select(f[i], fx.Float32(0.0)))
        lane = tx % 64
        ssq = wave_sum_lane63(ssq)
        wave = tx // 64
        if lane == 63:
            smem.ss.ptr[wave] = ssq
        fx.gpu.barrier()
        total = smem.ss.ptr[0] + smem.ss.ptr[1]
        rstd = rsq(total * fx.Float32(1.0 / N_FULL) + i32_eps.bitcast(fx.Float32))
        ores = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=fx.Int64(i32_rows) * (N_FULL * 2))
        for s in range_constexpr(WORLD):
            wf = wts[s]
            o = Vec.from_elements([pk_bf16(vals[s][2 * i] * rstd * wf[2 * i], vals[s][2 * i + 1] * rstd * wf[2 * i + 1])
                                   for i in range_constexpr(4)], fx.Int32)
            buffer_ops.buffer_store(o, ores, (row * N_FULL + s * N_SHARD + c * 8) // 2, mask=mine_ok)
        if tx == 0:
            buffer_ops.buffer_store(e + 1, cr, row)

    @flyc.jit
    def launch(i32_rows: fx.Int32, i32_me: fx.Int32, arg_slab: fx.Int64, arg_bufs: fx.Int64, arg_cnt: fx.Int64,
               arg_w: fx.Int64, arg_out: fx.Int64, i32_eps: fx.Int32, stream: fx.Stream):
        kernel(i32_me, i32_rows, arg_slab, arg_bufs, arg_cnt, arg_w, arg_out, i32_eps).launch(
            grid=(fx.Int64(i32_rows), 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
