# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank one-shot all-reduce with the 12-wave norm and the bf16-row quant rule.

Rows go in batches of ``rpb``; the grid is stagers, then reducers, then (``owner``) receivers, each role striding over
the batches. All-row form: a stager writes its partial rows into its own slots, a reducer reads the peer's over the
link, sums them with its own per row and adds, norms (and quantizes) every row. ``owner``: rows split into chunk c =
rows [c q, (c + 1) q), q = ceil(rows / 2), owned by rank c: a stager writes only the peer's chunk; a reducer reduces
its own chunk (residual read and written there only) and stages the result (6144 code + 192 scale bytes, or the bf16
row) in its own slots; a receiver copies the peer's results into its outputs. Stagers never wait and reducers wait only
for peer stagers, so with that dispatch order the grid needs no residency guarantee."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import COUNT_WORDS, RESULT_FLAG, SYS, flag_addr, flag_store, slot_addr, spin, sum2
from .norm_math import PRELOAD_HINTS, H, _block_sum, bf16x8, pk_bf16, rsqrt_rn, seq_sum_sq, swap_xor1, wave_sum_lane63
from .quant_math import _code_off, _codes, _e8m0, _scale_off, _set_fp16_ovfl

NW = H // 8 // 64  # 12 waves of 8 columns per lane (the plain norm's layout)
NT = 64 * NW


def compile_allreduce_gemma_bf16_mxfp8(rpb: int, shuffled: bool, quantized: bool, owner: bool = False):
    B = rpb
    name = (f"allreduce_gemma_bf16_{'mxfp8' if quantized else 'out'}_{'tiled' if shuffled else 'rowmajor'}"
            f"_h6144_ranks2_b{B}" + ("_owner" if owner else ""))

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 16]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_me: fx.Int32,
        i32_grid: fx.Int32,  # stagers | reducers << 10 | receivers << 20
        i32_sr: fx.Int32,  # residual row stride (elements)
        arg_x: fx.Int64,  # [m, H] bf16 partial, contiguous
        arg_mine: fx.Int64,
        arg_peer: fx.Int64,
        arg_cnt: fx.Int64,  # int32 [3, COUNT_WORDS]: stagers, reducers, receivers
        arg_r: fx.Int64,
        arg_w: fx.Int64,
        arg_q: fx.Int64,  # codes, or the bf16 out
        arg_s: fx.Int64,
        arg_res: fx.Int64,
        i32_eps: fx.Int32,
    ):
        if const_expr(quantized):
            _set_fp16_ovfl()
        tx, b = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        m, me = i32_m, i32_me
        nstage = i32_grid & 0x3FF
        nred = (i32_grid >> 10) & 0x3FF
        nrecv = i32_grid >> 20
        nb = (m + (B - 1)) // B
        q = (m + 1) // 2
        nb0 = (q + (B - 1)) // B
        if const_expr(owner):
            nb = nb0 + (m - q + (B - 1)) // B
        nbytes = H * 2
        off = tx * 4
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=3 * COUNT_WORDS * 4)

        def batch(k):
            """(first row, end of its chunk, its chunk) of batch k."""
            if const_expr(owner):
                c = (k >= nb0).select(fx.Int32(1), fx.Int32(0))
                r0 = (c == 0).select(k * B, q + (k - nb0) * B)
                return r0, (c == 0).select(q, m), c
            return k * B, m, fx.Int32(0)

        def count(role, r):
            return fx.Int32(buffer_ops.buffer_load(cr, role * COUNT_WORDS + r, vec_width=1, dtype=fx.Int32,
                                                   is_scalar=True)) + 1

        def advance(roles, first, step):
            """After the workgroup's last read: count + 1 for every row of its batches."""
            fx.gpu.barrier()
            for k in range(first, nb, step):
                r0, end, _c = batch(k)
                r = r0 + tx
                ok = (tx < B) & (r < end)
                for role in roles:
                    cur = fx.Int32(buffer_ops.buffer_load(cr, role * COUNT_WORDS + r, vec_width=1, dtype=fx.Int32,
                                                          mask=ok))
                    buffer_ops.buffer_store(cur + 1, cr, role * COUNT_WORDS + r, mask=ok)

        def read_row(base, cm, ok):
            rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=nbytes)
            return Vec(buffer_ops.buffer_load(rs, off, vec_width=4, dtype=fx.Int32, mask=ok, cache_modifier=cm))

        if const_expr(quantized):
            q_bytes = ((m + 15) // 16 * 16 if shuffled else m) * H
        else:
            q_bytes = m * nbytes
        s_bytes = (m + 31) // 32 * 32 * (H // 32)

        def store_q(codes, row):
            qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=q_bytes)
            if const_expr(shuffled):
                # (16, 16) layout: even lanes store their piece's low 8 B plus the odd lane's high 8 B as one 16 B
                # store; odd lanes idle
                recv = swap_xor1(codes)
                piece = Vec.from_elements([codes[0], codes[1], recv[0], recv[1]], fx.Int32)
                buffer_ops.buffer_store(piece, qr, _code_off(row, tx // 2) // 4, mask=(tx % 2) == 0)
            else:
                buffer_ops.buffer_store(Vec.from_elements(codes, fx.Int32), qr, (row * (H // 4)) + tx * 2)

        def store_s(sbyte, row, g):
            sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=s_bytes)
            buffer_ops.buffer_store(sbyte.to(fx.Int8), sr, _scale_off(row, g), offset_is_bytes=True)

        def store_out(v, row):
            qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=q_bytes)
            buffer_ops.buffer_store(v, qr, row * (H // 2) + off)

        if b < nstage:
            for k in range(b, nb, nstage):
                r0, end, c = batch(k)
                if (c != me) | fx.Boolean(not owner):
                    for i in range_constexpr(B):
                        r = r0 + i
                        ok = r < end
                        dst = buffer_ops.create_buffer_resource_from_addr(
                            slot_addr(arg_mine, r, count(0, r)), num_records_bytes=nbytes)
                        buffer_ops.buffer_store(read_row(arg_x + fx.Int64(r) * nbytes, 0, ok), dst, off, mask=ok,
                                                cache_modifier=SYS)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    if lane == 0:
                        flag_store(flag_addr(arg_peer, r0, wave), count(0, r0))
            advance((0,), b, nstage)
        elif b < nstage + nred:
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=nbytes)
            wv = bf16x8(Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32)))
            for k in range(b - nstage, nb, nred):
                r0, end, c = batch(k)
                if (c == me) | fx.Boolean(not owner):
                    oks = [(r0 + i) < end for i in range_constexpr(B)]
                    owns = [read_row(arg_x + fx.Int64(r0 + i) * nbytes, 0, oks[i]) for i in range_constexpr(B)]
                    rvs = [read_row(arg_r + fx.Int64(r0 + i) * fx.Int64(i32_sr * 2), 0, oks[i])
                           for i in range_constexpr(B)]
                    # a poll waits for the wave's outstanding vector loads: the batch's own rows and residuals
                    if lane == 0:
                        spin(flag_addr(arg_mine, r0, wave), count(1, r0))
                    es = [count(1, r0 + i) for i in range_constexpr(B)]
                    staged = [read_row(slot_addr(arg_peer, r0 + i, es[i]), SYS, oks[i]) for i in range_constexpr(B)]
                    for i in range_constexpr(B):
                        row = r0 + i
                        if row < end:
                            xv = bf16x8(sum2(owns[i], staged[i]))
                            rv = bf16x8(rvs[i])
                            s = [xv[n] + rv[n] for n in range_constexpr(8)]
                            resr = buffer_ops.create_buffer_resource_from_addr(
                                arg_res + fx.Int64(row) * nbytes, num_records_bytes=nbytes)
                            res = [pk_bf16(s[2 * n], s[2 * n + 1]) for n in range_constexpr(4)]
                            buffer_ops.buffer_store(Vec.from_elements(res, fx.Int32), resr, off)
                            acc = wave_sum_lane63(seq_sum_sq(s))
                            if lane == 63:
                                part[wave] = acc
                            fx.gpu.barrier()
                            rstd = rsqrt_rn(_block_sum([part[n] for n in range_constexpr(NW)]),
                                            i32_eps.bitcast(fx.Float32))
                            y = [(s[n] * rstd) * (wv[n] + fx.Float32(1.0)) for n in range_constexpr(8)]
                            ypack = Vec.from_elements([pk_bf16(y[2 * n], y[2 * n + 1]) for n in range_constexpr(4)],
                                                      fx.Int32)
                            result = buffer_ops.create_buffer_resource_from_addr(slot_addr(arg_mine, row, es[i]),
                                                                                 num_records_bytes=nbytes)
                            if const_expr(quantized):
                                # the standalone quantizer reads the bf16-rounded normed value
                                sb, rescale = _e8m0(bf16x8(ypack))
                                codes = _codes(bf16x8(ypack), rescale)  # 2 dwords, column order
                                sbyte = fx.Int32(sb.to(fx.Int32))
                                store_q(codes, row)
                                if (tx % 4) == 0:
                                    store_s(sbyte, row, tx // 4)
                                if const_expr(owner):
                                    # the staged result: 6144 code bytes, then 192 scale bytes
                                    buffer_ops.buffer_store(Vec.from_elements(codes, fx.Int32), result, tx * 2,
                                                            cache_modifier=SYS)
                                    buffer_ops.buffer_store(sbyte.to(fx.Int8), result, H + tx // 4,
                                                            mask=(tx % 4) == 0, offset_is_bytes=True,
                                                            cache_modifier=SYS)
                            else:
                                store_out(ypack, row)
                                if const_expr(owner):
                                    buffer_ops.buffer_store(ypack, result, off, cache_modifier=SYS)
                            fx.gpu.barrier()
                    if const_expr(owner):
                        fx.rocdl.s_waitcnt(vmcnt=0)
                        if lane == 0:
                            flag_store(flag_addr(arg_peer, r0, RESULT_FLAG + wave), count(1, r0))
            advance((1,) if owner else (1, 2), b - nstage, nred)
        else:
            for k in range(b - nstage - nred, nb, nrecv):
                r0, end, c = batch(k)
                if c != me:
                    if lane == 0:
                        spin(flag_addr(arg_mine, r0, RESULT_FLAG + wave), count(2, r0))
                    for i in range_constexpr(B):
                        row = r0 + i
                        ok = row < end
                        src = buffer_ops.create_buffer_resource_from_addr(slot_addr(arg_peer, row, count(2, row)),
                                                                          num_records_bytes=nbytes)
                        if const_expr(quantized):
                            cv = Vec(buffer_ops.buffer_load(src, tx * 2, vec_width=2, dtype=fx.Int32, mask=ok,
                                                            cache_modifier=SYS))
                            # lane 16 l copies the scale bytes of groups 4 l .. 4 l + 3 (its wave wrote them)
                            sw = fx.Int32(buffer_ops.buffer_load(src, H // 4 + tx // 16, vec_width=1, dtype=fx.Int32,
                                                                 mask=ok & ((tx % 16) == 0), cache_modifier=SYS))
                            if ok:
                                store_q([fx.Int32(cv[0]), fx.Int32(cv[1])], row)
                                if (tx % 16) == 0:
                                    for j in range_constexpr(4):
                                        store_s((sw >> (8 * j)) & 0xFF, row, tx // 4 + j)
                        else:
                            v = Vec(buffer_ops.buffer_load(src, off, vec_width=4, dtype=fx.Int32, mask=ok,
                                                           cache_modifier=SYS))
                            if ok:
                                store_out(v, row)
            advance((2,), b - nstage - nred, nrecv)

    @flyc.jit
    def launch_allreduce_gemma_bf16_mxfp8_ranks2(
        i32_m: fx.Int32, i32_me: fx.Int32, i32_grid: fx.Int32, i32_sr: fx.Int32, arg_x: fx.Int64,
        arg_mine: fx.Int64, arg_peer: fx.Int64, arg_cnt: fx.Int64, arg_r: fx.Int64, arg_w: fx.Int64,
        arg_q: fx.Int64, arg_s: fx.Int64, arg_res: fx.Int64, i32_eps: fx.Int32, stream: fx.Stream,
    ):
        kernel(i32_m, i32_me, i32_grid, i32_sr, arg_x, arg_mine, arg_peer, arg_cnt, arg_r, arg_w, arg_q, arg_s,
               arg_res, i32_eps).launch(
            grid=(fx.Int64((i32_grid & 0x3FF) + ((i32_grid >> 10) & 0x3FF) + (i32_grid >> 20)), 1, 1),
            block=(NT, 1, 1), stream=stream,
        )

    launch_allreduce_gemma_bf16_mxfp8_ranks2.kernel_name = name
    launch_allreduce_gemma_bf16_mxfp8_ranks2.compile_hints = PRELOAD_HINTS
    return launch_allreduce_gemma_bf16_mxfp8_ranks2
