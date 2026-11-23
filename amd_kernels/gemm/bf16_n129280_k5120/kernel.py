# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 5120] x bf16 [129280, 5120]^T -> bf16 [M, 129280] for 1 to 64 rows, fp32 accumulate (the lm_head).

Tiles: the 8080 16-row vocabulary groups go to 505 tiles of 16 groups (the grid is padded to 512 workgroups; the extra
ones repeat the last tile and store the same bits). A workgroup of 4 waves covers a tile and all rows of its M tile
(16 MT rows; wave w takes the groups w, w + 4, w + 8, w + 12). The slot ring is 2 blocks, so a workgroup holds at most
80 KB of LDS and two workgroups share a CU: the second one's prologue and epilogue overlap the first one's steady
state, and the kernel streams the weight at 6.5 TB/s and above.

Data path: per 64-K block, both operands go HBM / L2 -> LDS by asynchronous `buffer_load ... lds` (no VGPRs spent
on bytes in flight) into a ring of 2 slots. One slot = the tile's 16 weight fragments + the MT activation fragments,
2 KB each (16 rows x 128 B). One DMA moves 8 rows x a 128-byte line; lane l writes bytes [16 l, 16 l + 16) of its 1 KB,
and the global piece it fetches is XOR-swizzled so that the MFMA lane (row l16, K group g) reads its 16 bytes
conflict-free. The waves share the activation slot: before block b + 1 each wave waits for its own DMAs (vmcnt) and
its LDS reads (lgkmcnt 0), one `s_barrier`, then refills slot b (block b + 2). Half-block software pipeline: the
ds_reads of (b, K half 1) overlap the MFMAs of (b, half 0), those of (b + 1, half 0) the MFMAs of (b, half 1)
(v_mfma_f32_16x16x32_bf16, K in ascending order).
Weight loads are non-temporal (`nt`): the weight (1.32 GB) streams through.

Weight layouts (one tensor serves every M):
- ROW: the checkpoint's [129280, 5120] row-major weight;
- SHUF: `shuffle_weight`'s load-time layout [8080 groups, 80 K blocks, 16 rows, 64 K]: every DMA reads 1 KB in one
  piece and a group's bytes follow each other block after block (bit-exact transform, one copy).

Every output element is one wave's fp32 sum in ascending K (no split-K, no atomics), rounded once to bf16.
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

TILES = 505  # vocabulary tiles per M tile, 16 groups each
GRID_TILES = (TILES + 7) // 8 * 8  # workgroups per M tile: whole XCD sets; the extra ones repeat the last tile
G16 = N // 16  # 8080 16-row groups
NR = G16 // TILES
assert NR * TILES == G16 and NR == 16
NWV = 4  # waves
NRW = NR // NWV  # 4 groups per wave
NB = K // 64  # 80 K blocks
W_CM = 2  # nt
MT_MAX = 4  # (NR + MT) x 2 KB x 2 slots <= 80 KB: two workgroups per CU


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


def compile_bf16_gemm_n129280_k5120(mt: int, shuf: int):
    MT, SHUF = mt, shuf
    R = 2
    TM = 16 * MT
    XD = MT * 2  # activation DMAs per block (2 per 16-row fragment)
    XPW = (XD + NWV - 1) // NWV
    PER = NRW * 2 + XPW  # DMAs per wave per block
    SLOT = (NR + MT) * 2048
    LDS = R * SLOT
    assert 1 <= MT <= MT_MAX and 2 * LDS <= 160 * 1024, (LDS, MT)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    name = f"bf16_gemm_n129280_k5120_tm{TM}_r{R}" + ("_shuf" if SHUF else "_row")

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_m: fx.Int32, i32_mtiles: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave_all = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wave = wave_all % NWV  # the weight groups
        # workgroup b runs on XCD b % 8; the M tiles of one vocabulary tile follow each other on that XCD
        xcd = bid % 8
        j = bid // 8
        t = (j // i32_mtiles) * 8 + xcd
        m0 = (j % i32_mtiles) * TM
        t = (t < TILES).select(t, fx.Int32(TILES - 1))
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
            for jj in range_constexpr(2):
                pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
                if const_expr(SHUF):
                    off = (n0 // 16 + rn) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
                else:
                    off = (n0 + rn * 16 + jj * 8 + r8) * (K * 2) + pc * 16
                w_src.append(off)
                w_dst.append((rn * 2 + jj) * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave_all + i * NWV
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = m0 + (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)  # rows past M read the last row; never stored
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (K * 2) + px * 16)
            x_dst.append((NR * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % R) * SLOT + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def dma(b):
            for q in range_constexpr(NRW * 2):
                issue(wr, w_dst[q], w_src[q], b, b * (2048 if SHUF else 128), aux_w)
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, b * 128, aux_x)

        wpos = [wave + i * NWV for i in range_constexpr(NRW)]
        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % R) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)] for _ in range_constexpr(NRW)]

        def read_ops(b, kk):
            return ([rd(b, wpos[i], kk) for i in range_constexpr(NRW)],
                    [rd(b, NR + m, kk) for m in range_constexpr(MT)])

        def mfmas(ops):
            wv, xv = ops
            for i in range_constexpr(NRW):
                for m in range_constexpr(MT):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m], acc[i][m], 0, 0, 0]))

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        # half-block software pipeline: the LDS reads of (b, kk 1) overlap the MFMAs of (b, kk 0), those of
        # (b + 1, kk 0) the MFMAs of (b, kk 1); slot b is refilled (block b + R) once every wave has read it
        # (lgkmcnt(0) before the barrier), so R blocks are in flight
        for b in range_constexpr(R):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(R - 1) * PER)
        barrier()
        _sb()
        ops_a = read_ops(0, 0)
        for b in range_constexpr(NB):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < NB:
                # blocks issued so far: 0 .. min(b + R - 1, NB - 1); block b + 1 must have landed
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NB - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                ops_a = read_ops(b + 1, 0)
                _sb()
                if b + R < NB:
                    dma(b + R)
                _sb()
            mfmas(ops_b)

        for i in range_constexpr(NRW):
            rn = wave + i * NWV
            n = n0 + rn * 16 + g * 4
            for m in range_constexpr(MT):
                r = m0 + m * 16 + l16
                pk = acc[i][m].to(fx.BFloat16).bitcast(fx.Int32)
                buffer_ops.buffer_store(pk, yr, r * (N // 2) + n // 2, mask=r < i32_m)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, i32_m: fx.Int32, i32_mtiles: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, i32_m, i32_mtiles).launch(
            grid=(GRID_TILES * i32_mtiles, 1, 1), block=(64 * NWV, 1, 1), stream=stream)

    return launch
