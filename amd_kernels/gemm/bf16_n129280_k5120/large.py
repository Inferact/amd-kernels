# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 5120] x bf16 [129280, 5120]^T -> bf16 [M, 129280] for 65 to 2048 rows, fp32 accumulate: 8 waves per CU.

Tiles: the 8080 16-row vocabulary groups go to 505 tiles of 16 groups (the grid is padded to 512 workgroups; the extra
ones repeat the last tile and store the same bits); one workgroup covers a tile and 32 mt rows. Wave w takes the
groups w, w + 4, w + 8, w + 12 and the M half w // 4. The M tiles of one vocabulary tile run on one XCD, so the
weight tile comes from HBM once and from that XCD's L2 afterwards.

Data path as in kernel.py (asynchronous `buffer_load ... lds`, XOR-swizzled 128-byte lines, half-block software
pipeline, barrier per 64-K block), with two rings of their own depth: weights (HBM latency) and activations
(L2 latency). With one shared ring of 2 the DMAs of block b + 1 are issued one block of MFMAs before they are
needed and the kernel is latency bound, not MFMA bound; the table RINGS holds the depths that the 160 KB LDS allows
per M tile. With rotm = 80 each vocabulary tile starts its K loop at block t % 80 and wraps: the 505 workgroups then do
not request the same activation lines at the same time (up to 256 rows, whose 2.6 MB of activations stay in L2); with
rotm = 1 they run in lockstep (above 256 rows the rotated sweep's activation working set exceeds the 4 MB L2: -9 % at
640-768 rows). Every output element is still one wave's fp32 sum in a K order fixed by its vocabulary tile and rotm (no
split-K, no atomics), rounded once to bf16 (so its bits differ from kernel.py's in the last place); the epilogue stages the tile through the dead ring
to write 16-byte pieces.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from .reference import K, N

G16 = N // 16  # 8080 16-row groups
TILES = 505  # vocabulary tiles per M tile, 16 groups each
GRID_TILES = (TILES + 7) // 8 * 8  # workgroups per M tile: whole XCD sets; the extra ones repeat the last tile
NR = G16 // TILES
assert NR * TILES == G16 and NR == 16
NWV = 4
NRW = NR // NWV
NB = K // 64
W_CM = 2  # nt wins the full-shaped sweep; cached reads are slower
# (weight, activation) ring slots by M fragments per wave: the largest rings the 160 KB LDS holds that measure best
# (sweep: benchmarks/gemm/bf16_n129280_k5120/sweep_rings.py); a slot is NR x 2 KB of weights or 2 mt x 2 KB of activations
RINGS = {3: (3, 3), 4: (3, 3), 5: (3, 3), 6: (2, 4), 7: (2, 3), 8: (2, 3)}


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def compile_large_bf16_gemm_n129280_k5120(mt: int, shuf: int, rotm: int = NB):
    # Wave w owns vocabulary fragments w % 4 and M half w // 4.
    MT, SHUF = mt, shuf
    assert NB % rotm == 0, rotm
    NWT = NWV * 2  # waves per workgroup
    MTT = MT * 2  # activation fragments per workgroup
    TM = 16 * MTT
    XD = MTT * 2  # activation DMAs per block (2 per 16-row fragment)
    XPW = (XD + NWT - 1) // NWT
    PER = NRW + XPW  # DMAs per wave per block (each M half issues one of a group's two weight DMAs)
    RW, RX = RINGS[MT]
    WSLOT, XSLOT = NR * 2048, MTT * 2048
    XBASE = RW * WSLOT
    LDS = max(XBASE + RX * XSLOT, TM * 256 * 2)
    assert MT in RINGS and LDS <= 160 * 1024 and (max(RW, RX) - 1) * PER <= 63, (LDS, RW, RX, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"bf16_gemm_n129280_k5120_large_tm{TM}_r{RW}{RX}" + (f"_rot{rotm}" if rotm != NB else "") + ("_shuf" if SHUF else "_row")

    @flyc.kernel(name=name, known_block_size=[64 * NWT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_m: fx.Int32, i32_mtiles: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave_all = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wave = wave_all % NWV  # the weight groups
        mhalf = wave_all // NWV  # the M half
        # workgroup b runs on XCD b % 8; the M tiles of one vocabulary tile follow each other on that XCD
        xcd = bid % 8
        j = bid // 8
        t = (j // i32_mtiles) * 8 + xcd
        m0 = (j % i32_mtiles) * TM
        t = (t < TILES).select(t, fx.Int32(TILES - 1))
        rot = (t % rotm) * (NB // rotm)  # the K block this tile starts at (rotm distinct starts)
        n0 = t * (16 * NR)
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        yr = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_m * (N * 2))
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        # DMA jj of a fragment: rows 8 jj + r8, 16-byte piece (lane % 8) ^ ((r8 >> 1) | (jj << 2)) of the line
        w_src, w_dst = [], []
        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            jj = mhalf
            pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
            if const_expr(SHUF):
                off = (n0 // 16 + rn) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
            else:
                off = (n0 + rn * 16 + jj * 8 + r8) * (K * 2) + pc * 16
            w_src.append(off)
            w_dst.append((rn * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave_all + i * NWT
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = m0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)  # rows past M read the last row; never stored
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (K * 2) + px * 16)
            x_dst.append(d * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        log = []  # DMAs in issue order: (operand, block)

        def issue(rs, dst, src, b, soff, aux, operand):
            log.append((operand, b))
            base = (b % RW) * WSLOT if operand == "w" else XBASE + (b % RX) * XSLOT
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + base + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(fx.rocdl.readfirstlane(T.i32, ((rot + b) % NB) * soff)).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def dma(b):
            if b < RW:
                for q in range_constexpr(NRW):
                    issue(wr, w_dst[q], w_src[q], b, (2048 if SHUF else 128), aux_w, "w")
            if b < RX:
                for q in range_constexpr(XPW):
                    issue(xr, x_dst[q], x_src[q], b, 128, aux_x, "x")

        def landed(*need):
            """vmcnt that leaves only the DMAs issued after the last of the blocks in ``need``."""
            return len(log) - 1 - max(i for i, e in enumerate(log) if e in need)

        wpos = [wave + i * NWV for i in range_constexpr(NRW)]
        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(base, pos, kk):
            p = _lds_ptr(lds_base + base + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)] for _ in range_constexpr(NRW)]

        P = (MT + 1) // 2
        def read_w(b, kk):
            return [rd((b % RW) * WSLOT, wpos[i], kk) for i in range_constexpr(NRW)]
        def read_x(b, kk, lo, hi):
            return [rd(XBASE + (b % RX) * XSLOT, mhalf * MT + m, kk) for m in range_constexpr(lo, hi)]
        def mfmas(wv, xv, lo, hi, prefetch=None, operand=0):
            for i in range_constexpr(NRW):
                # Spread future DMA requests between four-MFMA groups instead
                # of issuing an eight-request burst before the computation.
                if prefetch is not None:
                    if operand == 0:
                        issue(wr, w_dst[i], w_src[i], prefetch, (2048 if SHUF else 128), aux_w, "w")
                    elif i < XPW:
                        issue(xr, x_dst[i], x_src[i], prefetch, 128, aux_x, "x")
                _sb()
                for m in range_constexpr(lo, hi):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m - lo], acc[i][m], 0, 0, 0]))

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        # half-block software pipeline: the LDS reads of (b, kk 1) overlap the MFMAs of (b, kk 0), those of
        # (b + 1, kk 0) the MFMAs of (b, kk 1); slot b is refilled (block b + R) once every wave has read it
        # (lgkmcnt(0) before the barrier), so R blocks are in flight
        for b in range_constexpr(max(RW, RX)):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=landed(("w", 0), ("x", 0)))
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
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=landed(("w", b + 1), ("x", b + 1)))
                barrier()
                _sb()
                w_a = read_w(b + 1, 0)
                x_a0 = read_x(b + 1, 0, 0, P)
                _sb()
                _sb()
            mfmas(w_b, x_b0, 0, P, (b + RW) if b + RW < NB else None, 0)
            _sb()
            if const_expr(b + 1 < NB):
                x_a1 = read_x(b + 1, 0, P, MT)
                _sb()
            mfmas(w_b, x_b1, P, MT, (b + RX) if b + RX < NB else None, 1)

        fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=0)
        barrier()
        out_view = fx.make_view(fx.recast_iter(fx.Int32, smem.buf.ptr), fx.make_layout(TM * 128, 1))
        out8 = fx.logical_divide(out_view, fx.make_layout(2, 1))
        out16 = fx.logical_divide(out_view, fx.make_layout(4, 1))
        atom8 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            for m in range_constexpr(MT):
                r = (mhalf * MT + m) * 16 + l16
                c = rn * 16 + g * 4
                pk = acc[i][m].to(fx.BFloat16).bitcast(fx.Int32)
                phys = (r * 128 + c // 2) ^ ((r % 8) * 4)
                rmem = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                rmem.store(pk)
                fx.copy(atom8, rmem, fx.slice(out8, (None, phys // 2)))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        for it in range_constexpr((TM * 256 // 8 + 64 * NWT - 1) // (64 * NWT)):
            v = tx + it * (64 * NWT)
            r = v // 32
            c = (v % 32) * 8
            phys = (r * 128 + c // 2) ^ ((r % 8) * 4)
            rmem = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom16, fx.slice(out16, (None, phys // 4)), rmem)
            val = Vec(rmem.load())
            buffer_ops.buffer_store(val, yr, (m0 + r) * (N // 2) + (n0 + c) // 2,
                                    mask=(r < TM) & (m0 + r < i32_m) & (c < 256))

    @flyc.jit
    def launch_large_bf16_gemm_n129280_k5120(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_m: fx.Int32, i32_mtiles: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, i32_m, i32_mtiles).launch(
            grid=(GRID_TILES * i32_mtiles, 1, 1), block=(64 * NWT, 1, 1), stream=stream)

    return launch_large_bf16_gemm_n129280_k5120
