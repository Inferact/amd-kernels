# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Five bf16 taps read in place, LDS-DMA ring for both operands, fp32 split-K slabs (1-1024 rows).

A workgroup owns ``NR`` 16-column weight fragments and ``MT * MH`` 16-row tap fragments over one K slab; the slab
never crosses a tap (``split`` is a multiple of 5 and 96 blocks of 64 K per tap divide evenly), so one tap buffer
serves the whole K loop. Wave w owns weight fragments w % 4 + 4 i and M half w // 4. Every 64-K block lands in an
LDS ring slot by async buffer->LDS DMA (weight in the packed layout, 2 KB per fragment, read back in lane order;
taps row-major with a 16-byte XOR swizzle), ``ring`` blocks in flight. Each slab keeps the ascending-K MFMA order.
Based on gemm/bf16_n50016_k6144/large.py.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

N, K, H = 3072, 30720, 6144
NWV = 4
TAP_BLOCKS = H // 64
W_CM = 2  # non-temporal weight DMAs: every weight byte is read by one M tile at a time


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def compile_concat_linear_lds(mt, nr, split, mh=2, ring=2):
    MT, NR, MH, R = mt, nr, mh, ring
    NB = K // 64 // split  # 64-K blocks per slab
    assert TAP_BLOCKS % NB == 0 and NR % NWV == 0 and (N // 16) % NR == 0, (NB, NR)
    NRW = NR // NWV
    NWT = NWV * MH
    MTT = MT * MH
    TM = 16 * MTT
    NTILES = N // (16 * NR)
    XD = MTT * 2
    XPW = (XD + NWT - 1) // NWT
    PER = NRW * 2 // MH + XPW
    SLOT = (NR + MTT) * 2048
    LDS = R * SLOT
    assert MH in (1, 2) and LDS <= 160 * 1024 and (R - 1) * PER <= 63, (LDS, R, PER)
    name = f"bf16_concat_linear_n3072_k30720_lds_mt{MT}_mh{MH}_nr{NR}_r{R}_sk{split}"

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NWT, 1, 1])
    def kernel(x0: fx.Int64, x1: fx.Int64, x2: fx.Int64, x3: fx.Int64, x4: fx.Int64,
               weight: fx.Int64, partials: fx.Int64, rows: fx.Int32,
               sx0: fx.Int32, sx1: fx.Int32, sx2: fx.Int32, sx3: fx.Int32, sx4: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave_all = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wave = wave_all % NWV
        mhalf = wave_all // NWV
        # XCD = bid % 8: the M tiles of one (column tile, slab), which share its weight slab, share an XCD when
        # split x column tiles is a multiple of 8
        s = bid % split
        nt = (bid // split) % NTILES
        mb = bid // (split * NTILES)
        m0 = mb * TM
        kb0 = s * NB
        tap = kb0 // TAP_BLOCKS
        xbase = (tap == 0).select(x0, (tap == 1).select(x1, (tap == 2).select(x2, (tap == 3).select(x3, x4))))
        sx = (tap == 0).select(sx0, (tap == 1).select(sx1, (tap == 2).select(sx2, (tap == 3).select(sx3, sx4))))
        xsoff = (kb0 % TAP_BLOCKS) * 128
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(xbase, num_records_bytes=fx.Int64(rows) * fx.Int64(sx * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(weight, num_records_bytes=N * K * 2)
        pr = buffer_ops.create_buffer_resource_from_addr(partials, num_records_bytes=fx.Int64(rows) * (N * split * 4))
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        w_src, w_dst = [], []
        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            for jq in range_constexpr(2 // MH):
                jj = mhalf + jq if MH == 2 else jq
                w_src.append((nt * NR + rn) * (K * 32) + (kb0 * 2 + jj) * 1024 + lane * 16)
                w_dst.append((rn * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave_all + i * NWT
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = m0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < rows).select(row, rows - 1)  # rows past M read the last row; never stored
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (sx * 2) + px * 16)
            x_dst.append((NR * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % R) * SLOT + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def issue_w(q, b):
            issue(wr, w_dst[q], w_src[q], b, b * 2048, aux_w)

        def issue_x(q, b):
            issue(xr, x_dst[q], x_src[q], b, xsoff + b * 128, aux_x)

        def dma(b):
            for q in range_constexpr(NRW * 2 // MH):
                issue_w(q, b)
            for q in range_constexpr(XPW):
                issue_x(q, b)

        r8r = l16 % 8
        hh = l16 // 8
        x_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(b, byte_off):
            p = _lds_ptr(lds_base + (b % R) * SLOT + byte_off)
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)] for _ in range_constexpr(NRW)]
        P = (MT + 1) // 2

        def read_w(b, kk):
            return [rd(b, ((wave + i * NWV) * 2 + kk) * 1024 + lane * 16) for i in range_constexpr(NRW)]

        def read_x(b, kk, lo, hi):
            return [rd(b, (NR + mhalf * MT + m) * 2048 + x_off[kk]) for m in range_constexpr(lo, hi)]

        def mfmas(wv, xv, lo, hi, prefetch=None, operand=0):
            for i in range_constexpr(NRW):
                # spread the next DMAs between MFMA groups instead of one burst before the computation
                if prefetch is not None:
                    if operand == 0:
                        for jq in range_constexpr(2 // MH):
                            issue_w(i * (2 // MH) + jq, prefetch)
                    elif i < XPW:
                        issue_x(i, prefetch)
                _sb()
                for m in range_constexpr(lo, hi):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m - lo], acc[i][m], 0, 0, 0]))
            if prefetch is not None and operand == 1:
                for q in range_constexpr(NRW, XPW):
                    issue_x(q, prefetch)

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        # half-block software pipeline (kk 0 / 1 of each 64-K block); slot b is refilled with block b + R once every
        # wave has read it (lgkmcnt(0) before the barrier)
        for b in range_constexpr(min(R, NB)):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(min(R, NB) - 1) * PER)
        barrier()
        _sb()
        w_a = read_w(0, 0)
        x_a0 = read_x(0, 0, 0, P)
        x_a1 = read_x(0, 0, P, MT)
        for b in range_constexpr(NB):
            _sb()
            w_b = read_w(b, 1)
            x_b0 = read_x(b, 1, 0, P)
            _sb()
            mfmas(w_a, x_a0, 0, P)
            _sb()
            x_b1 = read_x(b, 1, P, MT)
            _sb()
            mfmas(w_a, x_a1, P, MT)
            _sb()
            if const_expr(b + 1 < NB):
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NB - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                w_a = read_w(b + 1, 0)
                x_a0 = read_x(b + 1, 0, 0, P)
                _sb()
            mfmas(w_b, x_b0, 0, P, (b + R) if b + R < NB else None, 0)
            _sb()
            if const_expr(b + 1 < NB):
                x_a1 = read_x(b + 1, 0, P, MT)
                _sb()
            mfmas(w_b, x_b1, P, MT, (b + R) if b + R < NB else None, 1)

        for i in range_constexpr(NRW):
            for m in range_constexpr(MT):
                r = m0 + (mhalf * MT + m) * 16 + l16
                c = (nt * NR + wave + i * NWV) * 16 + g * 4
                buffer_ops.buffer_store(acc[i][m], pr, (s * rows + r) * N + c, mask=r < rows)

    @flyc.jit
    def launch_bf16_concat_linear_lds_n3072_k30720(x0: fx.Int64, x1: fx.Int64, x2: fx.Int64, x3: fx.Int64,
                                                   x4: fx.Int64, weight: fx.Int64, partials: fx.Int64,
                                                   rows: fx.Int32, sx0: fx.Int32, sx1: fx.Int32, sx2: fx.Int32,
                                                   sx3: fx.Int32, sx4: fx.Int32, stream: fx.Stream):
        kernel(x0, x1, x2, x3, x4, weight, partials, rows, sx0, sx1, sx2, sx3, sx4).launch(
            grid=(fx.Int64(((rows + TM - 1) // TM) * NTILES * split), 1, 1), block=(64 * NWT, 1, 1), stream=stream)

    launch_bf16_concat_linear_lds_n3072_k30720.kernel_name = name
    return launch_bf16_concat_linear_lds_n3072_k30720
