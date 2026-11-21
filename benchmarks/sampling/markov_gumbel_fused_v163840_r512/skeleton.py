# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Ceiling kernels of the benchmark: a read-only stream of the 168 MB ``markov_w2`` with the fused kernel's access pattern
(256 workgroups x 4 waves, one 16-column x 512 tile of 16 loads per wave and step, one tile ahead, XOR-reduced), no math,
no dependence between passes. ``passes`` = 1 is the per-launch floor of one Markov step (k launches per chain);
``passes`` = k is a persistent chain's floor without any step synchronization."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops
from amd_kernels.sampling.markov_gumbel_fused_v163840_r512 import RANK, V

I = fx.Int32
NWG = 256
NW = 4
TILES = V // 16 // (NWG * NW)


@functools.cache
def stream_exe(passes: int):
    @flyc.kernel(name=f"markov_w2_stream_p{passes}", known_block_size=[64 * NW, 1, 1])
    def kernel(arg_w: fx.Int64, arg_o: fx.Int64):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        w_r = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=V * RANK * 2)
        o_r = buffer_ops.create_buffer_resource_from_addr(arg_o, num_records_bytes=0x7FFFFFFF)
        n0 = bid * (TILES * NW * 16) + wave * TILES * 16

        def load(t):
            wi = n0 + t * 16 + lane % 16
            return [Vec(buffer_ops.buffer_load(w_r, wi * (RANK // 2) + k * 16 + (lane // 16) * 4, vec_width=4,
                                               dtype=I)) for k in range_constexpr(RANK // 32)]

        acc = Vec.filled(4, 0, I)
        cur = load(0)
        total = passes * TILES
        for g in range_constexpr(total):
            nxt = load((g + 1) % TILES) if g + 1 < total else cur
            for v in cur:
                acc = acc ^ v
            cur = nxt
            if (g + 1) % TILES == 0 and g + 1 < total:
                fx.gpu.barrier()  # a pass is a step of the chain: the barrier also keeps the compiler from merging its loads
        buffer_ops.buffer_store(acc, o_r, (bid * 64 * NW + tx) * 4)

    @flyc.jit
    def launch(arg_w: fx.Int64, arg_o: fx.Int64, stream: fx.Stream):
        kernel(arg_w, arg_o).launch(grid=(NWG, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    return launch


def run_stream(passes, w2, out, stream):
    exe = stream_exe(passes)
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, w2.data_ptr(), out.data_ptr(), stream)
    else:
        cf(w2.data_ptr(), out.data_ptr(), stream)
