# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The five aux hidden-state tensors [rows, 7168] -> the packed activation tiles of the projection (replaces
``torch.cat(aux, -1, out=staging)``). One wave per (16-row tile, 32-K step): its 64 lanes read 16 rows x 64 B and write
one contiguous 1 KB block. Rows past ``rows`` repeat the last row (their outputs are discarded)."""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .device import PRELOAD_HINTS, vload
from .reference import K

KSTEPS = K // 32
LAYERS = 5
HIDDEN = K // LAYERS
LAYER_STEPS = HIDDEN // 32
WAVES = 4


@functools.cache
@launcher
def compile_pack_rows():
    name = "bf16_pack_rows_k35840"

    @flyc.kernel(name=name, known_block_size=[64 * WAVES, 1, 1])
    def kernel(arg_out: fx.Int64, arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, arg_p3: fx.Int64,
               arg_p4: fx.Int64, i32_rows: fx.Int32, i32_stride: fx.Int32, i32_blocks: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        blk = bid * WAVES + tx // 64
        live = blk < i32_blocks
        blk = live.select(blk, fx.Int32(0))
        tile = blk // KSTEPS
        kstep = blk % KSTEPS
        layer = kstep // LAYER_STEPS
        row = tile * 16 + lane % 16
        row = (row < i32_rows).select(row, i32_rows - 1)
        col = (kstep % LAYER_STEPS) * 32 + (lane // 16) * 8
        base = (layer == 0).select(arg_p0, (layer == 1).select(arg_p1, (layer == 2).select(
            arg_p2, (layer == 3).select(arg_p3, arg_p4))))
        v = vload(base + (fx.Int64(row) * i32_stride + col) * 2)
        res = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=fx.Int64(i32_blocks) * 1024)
        buffer_ops.buffer_store(v, res, (blk * 64 + lane) * 4, mask=live)

    @flyc.jit
    def launch(arg_out: fx.Int64, arg_p0: fx.Int64, arg_p1: fx.Int64, arg_p2: fx.Int64, arg_p3: fx.Int64,
               arg_p4: fx.Int64, i32_rows: fx.Int32, i32_stride: fx.Int32, i32_blocks: fx.Int32,
               stream: fx.Stream):
        kernel(arg_out, arg_p0, arg_p1, arg_p2, arg_p3, arg_p4, i32_rows, i32_stride, i32_blocks).launch(
            grid=(fx.Int64((i32_blocks + WAVES - 1) // WAVES), 1, 1), block=(64 * WAVES, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
