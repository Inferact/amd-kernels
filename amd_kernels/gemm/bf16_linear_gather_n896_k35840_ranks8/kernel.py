# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Split-K bf16 GEMM of an N-shard (896 rows) of the K 35840 projection for up to 192 activation rows (gfx950).

Both operands are in the load-time MFMA layout (``pack.py``): one (16 rows, 32 K) tile is 1 KB contiguous in lane
order, so every wave load of weights or activations is one coalesced 1 KB. The weights are the MFMA's first operand,
the activation rows the second. A workgroup of ``wr * wk`` waves owns ``mt`` activation row tiles, ``g`` weight groups
of 16 rows and 1 / ``split`` of K: its waves split the row tiles (``wr``) and K (``wk``), so every activation and
weight byte is loaded once per workgroup. The K loop is fully unrolled with ``depth`` steps of loads in flight (a
loop-carried software pipeline drains its loads at every back edge). The waves of a K group are summed in wave order
through LDS, and the workgroup stores one fp32 slab [split, rows, 896] tile; the slabs are summed in slab order by
``reduce_gather.py``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import K, N_SHARD

KSTEPS = K // 32
GROUPS = N_SHARD // 16
LDS_MAX = 160 * 1024
NT = 0x2  # non-temporal weights keep the activation tiles in L2: 80 rows 17.2 -> 16.3 us (graph loop)


def _lds_ptr(addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(addr)))


@functools.cache
@launcher
def compile_linear(mt: int, g: int, split: int, depth: int, wr: int = 1, wk: int = 4):
    assert GROUPS % g == 0 and mt % wr == 0
    nw = wr * wk
    mtw = mt // wr
    nblocks = GROUPS // g
    steps = -(-KSTEPS // (split * wk))
    tw = mtw * g
    lds = nw * tw * 1024
    assert lds <= LDS_MAX, lds
    name = f"bf16_linear_n896_k35840_mt{mt}_g{g}_sk{split}_d{depth}_w{wr}x{wk}"

    def mfma(a, b, c):
        return Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(4, fx.Float32), [a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), c, 0, 0, 0]))

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, lds, 16]

    @flyc.kernel(name=name, known_block_size=[64 * nw, 1, 1])
    def kernel(x: fx.Int64, weight: fx.Int64, slab: fx.Int64, rows: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wi = wave % wr
        wj = wave // wr
        s = bid % split
        nb = (bid // split) % nblocks
        mb = bid // (split * nblocks)
        wres = buffer_ops.create_buffer_resource_from_addr(weight, num_records_bytes=N_SHARD * K * 2)
        sres = buffer_ops.create_buffer_resource_from_addr(slab, num_records_bytes=fx.Int64(rows) * (N_SHARD * split * 4))
        xres = buffer_ops.create_buffer_resource_from_addr(x, num_records_bytes=fx.Int64(0x7FFFFFF0))
        k0 = (s * wk + wj) * steps

        def issue(step):
            gs = k0 + step
            ok = gs < KSTEPS
            av = [Vec(buffer_ops.buffer_load(wres, ((nb * g + j) * KSTEPS + gs) * 256 + lane * 4, vec_width=4,
                                             dtype=fx.Int32, mask=ok, cache_modifier=NT)) for j in range_constexpr(g)]
            bv = [Vec(buffer_ops.buffer_load(xres, (((mb * mt + wi * mtw + i) * KSTEPS + gs) * 64 + lane) * 4,
                                             vec_width=4, dtype=fx.Int32, mask=ok)) for i in range_constexpr(mtw)]
            return av, bv

        acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(tw)]
        pend = {}
        for st in range_constexpr(min(depth, steps)):
            pend[st] = issue(st)
        for st in range_constexpr(steps):
            av, bv = pend.pop(st)
            if st + depth < steps:
                pend[st + depth] = issue(st + depth)
            fx.rocdl.sched_barrier(0)
            for i in range_constexpr(mtw):
                for j in range_constexpr(g):
                    acc[i * g + j] = mfma(av[j], bv[i], acc[i * g + j])
            fx.rocdl.sched_barrier(0)
        for t in range_constexpr(tw):
            _llvm.store(fx.as_ir_value(acc[t]), _lds_ptr(lds_base + ((wave * tw + t) * 64 + lane) * 16), alignment=16)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()
        items = wr * tw * 64
        for u in range_constexpr((items + 64 * nw - 1) // (64 * nw)):
            it = tx + u * 64 * nw
            ok = it < items
            itc = ok.select(it, fx.Int32(0))
            r = itc // (tw * 64)
            t = (itc // 64) % tw
            ln = itc % 64
            tot = Vec.filled(4, 0.0, fx.Float32)
            for w in range_constexpr(wk):
                tot = tot + Vec(_llvm.load(T.vec(4, T.f32),
                                           _lds_ptr(lds_base + (((w * wr + r) * tw + t) * 64 + ln) * 16), alignment=16))
            row = (mb * mt + r * mtw + t // g) * 16 + ln % 16
            col = (nb * g + t % g) * 16 + (ln // 16) * 4
            buffer_ops.buffer_store(tot, sres, (s * rows + row) * N_SHARD + col, mask=ok & (row < rows))

    @flyc.jit
    def launch(x: fx.Int64, weight: fx.Int64, slab: fx.Int64, rows: fx.Int32, stream: fx.Stream):
        kernel(x, weight, slab, rows).launch(
            grid=(fx.Int64(((rows + mt * 16 - 1) // (mt * 16)) * nblocks * split), 1, 1), block=(64 * nw, 1, 1),
            stream=stream)

    launch.kernel_name = name
    return launch
