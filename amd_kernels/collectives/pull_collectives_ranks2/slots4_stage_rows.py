# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Slot-order FP32 sum + shared output into a stage buffer (the SDMA hand-offs' reduce-scatter source).

256 threads x 8 BF16 values per thread, a grid-stride loop over 256 workgroups. The 16-byte pieces never cross a
6144-column row. The device counter's parity selects one of two slots ``i64_stride`` apart (0: one buffer).
``owner``: ``shared`` is [rows / 2, H], rank ``i32_me``'s shared expert output of its own block, added there only (the
other rows add +0 by a masked load).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import bf16x8, pk_bf16

H = 6144
NT = 256
WGS = 256  # the winner of a 128/256/512/1024/2048 row-stage sweep


def compile_slots4_stage_rows(owner: bool = False):
    name = "slots4_shared_stage_rows_h6144_ranks2" + ("_owner" if owner else "")
    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_slots: fx.Int64, arg_shared: fx.Int64, arg_pool: fx.Int64,
               counter: fx.Tensor, i64_stride: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32):
        call = fx.Int32(counter[0])
        dst = buffer_ops.create_buffer_resource_from_addr(
            arg_pool + fx.Int64(call & 1) * i64_stride,
            num_records_bytes=fx.Int64(i32_rows) * H * 2)
        m = i32_rows // 2
        hr = buffer_ops.create_buffer_resource_from_addr(
            arg_shared, num_records_bytes=fx.Int64(m if owner else i32_rows) * H * 2)
        piece = fx.block_idx.x
        while piece < i32_rows * 3:
            row = piece // 3
            col = (piece % 3) * (NT * 4) + fx.thread_idx.x * 4
            off = piece * (NT * 4) + fx.thread_idx.x * 4
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_slots + fx.Int64(row) * (4 * H * 2), num_records_bytes=4 * H * 2)
            vals = [bf16x8(Vec(buffer_ops.buffer_load(
                sr, col + k * (H // 2), vec_width=4, dtype=fx.Int32)))
                for k in range_constexpr(4)]
            if const_expr(owner):
                shared = bf16x8(Vec(buffer_ops.buffer_load(hr, off - i32_me * m * (H // 2), vec_width=4,
                                                           dtype=fx.Int32, mask=row // m == i32_me)))
            else:
                shared = bf16x8(Vec(buffer_ops.buffer_load(hr, off, vec_width=4, dtype=fx.Int32)))
            acc = [(((vals[0][j] + vals[1][j]) + vals[2][j]) + vals[3][j]) + shared[j]
                   for j in range_constexpr(8)]
            packed = [pk_bf16(acc[2*j], acc[2*j+1]) for j in range_constexpr(4)]
            buffer_ops.buffer_store(Vec.from_elements(packed, fx.Int32), dst, off)
            piece = piece + WGS

    @flyc.jit
    def launch_slots4_stage_rows(arg_slots: fx.Int64, arg_shared: fx.Int64, arg_pool: fx.Int64,
               counter: fx.Tensor, i64_stride: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32, stream: fx.Stream):
        kernel(arg_slots, arg_shared, arg_pool, counter, i64_stride, i32_rows, i32_me).launch(
            grid=(WGS, 1, 1), block=(NT, 1, 1), stream=stream)
    launch_slots4_stage_rows.kernel_name = name
    return launch_slots4_stage_rows
