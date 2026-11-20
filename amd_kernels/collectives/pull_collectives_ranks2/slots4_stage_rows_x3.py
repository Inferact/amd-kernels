# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Slot-order FP32 sum + shared output of every token, rounded once to BF16 (``slots4_stage_rows``'s sum): this rank's
rows as BF16 into ``own``, the peer's rows in ``lossless_x3``'s code into ``coded`` (the SDMA push source), their raw
virtual waves straight into ``raw`` (the peer's receive region, system-coherent stores over the link).

One wave per virtual wave (512 columns of a row) and step, a grid-stride loop over 256 workgroups of 4 waves.
``owner``: ``shared`` is [rows / 2, H], this rank's shared expert output of its own rows, added to them only (the
peer's rows add +0 by a masked load).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr
from flydsl.expr.typing import T

from .lossless_x3 import CODED, ROWB, VWAVES, enc_store
from .slots4_stage_pull_sum2_add_rmsnorm_mxfp8 import _slot_sum8

NT = 256
WGS = 256


def compile_slots4_stage_rows_x3(owner: bool = False):
    name = "slots4_shared_stage_rows_x3_h6144_ranks2" + ("_owner" if owner else "")

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_slots: fx.Int64, arg_shared: fx.Int64, arg_own: fx.Int64, arg_coded: fx.Int64,
               arg_raw: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32):
        m = i32_rows // 2
        lane = fx.thread_idx.x % 64
        it = fx.block_idx.x * (NT // 64) + fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.thread_idx.x // 64))
        own = buffer_ops.create_buffer_resource_from_addr(arg_own, num_records_bytes=fx.Int64(m) * ROWB)
        while it < i32_rows * VWAVES:
            row = it // VWAVES
            tid = (it % VWAVES) * 64 + lane
            r = row - (row >= m).select(m, fx.Int32(0))
            mine = (row >= m) == (i32_me == 1)
            if const_expr(owner):
                val = _slot_sum8(arg_slots, arg_shared, row, tid, shared_row=r, shared_mask=mine)
            else:
                val = _slot_sum8(arg_slots, arg_shared, row, tid)
            if mine:
                buffer_ops.buffer_store(val, own, r * (ROWB // 4) + tid * 4)
            else:
                enc_store(val, arg_coded + fx.Int64(r) * CODED, arg_raw + fx.Int64(r) * ROWB, tid)
            it = it + WGS * (NT // 64)

    @flyc.jit
    def launch_slots4_stage_rows_x3(arg_slots: fx.Int64, arg_shared: fx.Int64, arg_own: fx.Int64,
                                    arg_coded: fx.Int64, arg_raw: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32,
                                    stream: fx.Stream):
        kernel(arg_slots, arg_shared, arg_own, arg_coded, arg_raw, i32_rows, i32_me).launch(
            grid=(WGS, 1, 1), block=(NT, 1, 1), stream=stream)

    launch_slots4_stage_rows_x3.kernel_name = name
    return launch_slots4_stage_rows_x3
