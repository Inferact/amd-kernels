# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``sum2_add_rmsnorm_mxfp8`` with the peer's slab in ``lossless_x3``'s code: this rank's BF16 rows, the peer's coded
rows and their raw virtual waves (``slots4_stage_rows_x3``), decoded in the norm lane map, then the same rank-order
sum, residual add, norm and quant: the same output bits."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.add_rmsnorm_mxfp8 import NW, _block_sum, _set_fp16_ovfl
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import (
    PRELOAD_HINTS, H, bf16x8, pk_bf16, rsqrt_rn, wave_sum_lane63, xcd_grid, xcd_row,
)
from .lossless_x3 import CODED, ROWB, dec16, load16
from .sum2_add_rmsnorm_mxfp8 import _tail_rowq


def _ld(rs, tx, v, has):
    """Thread tx's 8 columns 16 tx + 8 v .. of a bf16 row as 4 dwords."""
    return Vec(buffer_ops.buffer_load(rs, tx * 8 + v * 4, vec_width=4, dtype=fx.Int32, mask=has))


def compile_sum2_x3_add_rmsnorm_mxfp8():
    name = "sum2_x3_add_rmsnorm_mxfp8_rowmajor_h6144_ranks2"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, NW]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_me: fx.Int32,
        arg_own: fx.Int64,  # this rank's [m, H] BF16 rows
        arg_coded: fx.Int64,  # the peer's [m, CODED] coded rows
        arg_raw: fx.Int64,  # their raw virtual waves, [m, H] BF16 slots
        arg_r: fx.Int64,  # residual [m, H] bf16
        arg_w: fx.Int64,  # norm weight [H] bf16
        arg_res: fx.Int64,  # residual out [m, H] bf16
        arg_q: fx.Int64,  # codes e4m3 [m, H] row-major
        arg_s: fx.Int64,  # scales u8, AITER layout
        i64_s_bytes: fx.Int64,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        _set_fp16_ovfl()
        tx = fx.thread_idx.x
        row, live = xcd_row(fx.block_idx.x, i32_m)
        if live:
            has = tx < 384
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            nb = H * 2
            recs, hw = load16(arg_coded + fx.Int64(row) * CODED, tx, has, 0)
            ownr = buffer_ops.create_buffer_resource_from_addr(arg_own + fx.Int64(row) * nb, num_records_bytes=nb)
            rr = buffer_ops.create_buffer_resource_from_addr(arg_r + fx.Int64(row) * nb, num_records_bytes=nb)
            wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=nb)
            own = [_ld(ownr, tx, v, has) for v in range_constexpr(2)]
            rv = bf16x8(_ld(rr, tx, 0, has)) + bf16x8(_ld(rr, tx, 1, has))
            wv = bf16x8(_ld(wr, tx, 0, has)) + bf16x8(_ld(wr, tx, 1, has))
            peer = dec16(recs, hw, arg_raw + fx.Int64(row) * ROWB, tx, has, 0)
            first = i32_me == 0
            b0 = []
            b1 = []
            for v in range_constexpr(2):
                for i in range_constexpr(4):
                    o = fx.Int32(own[v][i])
                    p = fx.Int32(peer[v][i])
                    b0.append(first.select(o, p))
                    b1.append(first.select(p, o))
            x0 = bf16x8(b0[:4]) + bf16x8(b0[4:])
            x1 = bf16x8(b1[:4]) + bf16x8(b1[4:])
            acc = [(fx.Float32(0.0) + x0[i]) + x1[i] for i in range_constexpr(16)]
            red = [pk_bf16(acc[2 * i], acc[2 * i + 1]) for i in range_constexpr(8)]
            xv = bf16x8(red[:4]) + bf16x8(red[4:])
            s = [xv[i] + rv[i] for i in range_constexpr(16)]
            res = [pk_bf16(s[2 * i], s[2 * i + 1]) for i in range_constexpr(8)]
            resr = buffer_ops.create_buffer_resource_from_addr(arg_res + fx.Int64(row) * nb, num_records_bytes=nb)
            for v in range_constexpr(2):
                buffer_ops.buffer_store(Vec.from_elements(res[v * 4:v * 4 + 4], fx.Int32), resr, tx * 8 + v * 4,
                                        mask=has)
            fma = lambda a, b2: fx.Float32(fx.math.fma(a, a, b2 * b2))  # noqa: E731
            e = [fma(s[4 * i], s[4 * i + 2]) for i in range_constexpr(4)]
            o = [fma(s[4 * i + 1], s[4 * i + 3]) for i in range_constexpr(4)]
            ssq = ((e[0] + e[1]) + (e[2] + e[3])) + ((o[0] + o[1]) + (o[2] + o[3]))
            acc2 = wave_sum_lane63(ssq)
            if lane == 63:
                part[wave] = acc2
            fx.gpu.barrier()
            rstd = rsqrt_rn(_block_sum([part[i] for i in range_constexpr(NW)]), i32_eps.bitcast(fx.Float32))
            qr = buffer_ops.create_buffer_resource_from_addr(arg_q, num_records_bytes=fx.Int64(i32_m) * H)
            sr = buffer_ops.create_buffer_resource_from_addr(arg_s, num_records_bytes=i64_s_bytes)
            _tail_rowq(tx, row, lane, s, wv, rstd, has, qr, sr)

    @flyc.jit
    def launch_sum2_x3_add_rmsnorm_mxfp8(
        arg_own: fx.Int64, arg_coded: fx.Int64, arg_raw: fx.Int64, arg_r: fx.Int64, arg_w: fx.Int64,
        arg_q: fx.Int64, arg_s: fx.Int64, arg_res: fx.Int64, i64_s_bytes: fx.Int64, i32_m: fx.Int32,
        i32_me: fx.Int32, i32_eps: fx.Int32, stream: fx.Stream,
    ):
        kernel(i32_m, i32_me, arg_own, arg_coded, arg_raw, arg_r, arg_w, arg_res, arg_q, arg_s, i64_s_bytes,
               i32_eps).launch(grid=(fx.Int64(xcd_grid(i32_m)), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch_sum2_x3_add_rmsnorm_mxfp8.kernel_name = name
    launch_sum2_x3_add_rmsnorm_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_sum2_x3_add_rmsnorm_mxfp8
