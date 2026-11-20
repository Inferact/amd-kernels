# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Persistent 4-wave MXFP8 GEMM for N4992 K6144, row-major codes and shuffled scales.

The pipeline of ``mxfp8_common``'s 4-wave kernel with a tile that divides N 4992
(64 n_a rows x 64 n_b columns, n_b 2 or 3: 39 or 26 column tiles) and its own tile
grid, so no work item reaches a column past N (the common kernel's 256-column parents
would wrap the last 128 columns into the next row). Quadrants of an odd tile count
re-pair E8M0 words across their 32-row scale groups; scale DMA copies contiguous
128-row packets. A 48-column quadrant row (n_b 3) leaves the output through an
unswizzled LDS transpose of 16-byte pieces, two stores per 16-row block. Every
accumulator visits K in increasing order, one 16x16x128 MFMA per step, so every tile
(and a launch over a row range) writes the same bytes: the host picks tiles per row
count and splits a call's rows over two tiles on that.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from amd_kernels.gemm.mxfp8_common.mxfp8_gemm_4wave import (
    BLOCK_K,
    G2SLoaderAsm,
    MfmaRowStationary,
    S2RLoader,
    StoreCLds as StoreCLdsBase,
    _Buf,
    _LDS_BUF_BYTES,
    _N_WAVES,
    _PACK,
    _SCALE_A_REGION,
    _SCALE_B_REGION,
    _SCALE_SLOTS,
    _SCALE_SLOT_BYTES,
    _ceildiv,
    _g2s_thunks,
    _global_swizzle,
    _lds_ptr_t,
    _min,
    _riffle,
    _s2r_thunks,
    _uniform_i32,
    _xcd_swizzle,
    wait_barrier,
    ScaleGatherLDS as ScaleGatherBase,
)


class ScaleGatherLDS(ScaleGatherBase):
    def tile_group(self, a_base_tile, b_base_tile):
        """This wave's first 32-row group of a tile (wave-uniform)."""
        base_tile = fx.arith.select(
            self._is_a,
            a_base_tile + self._q * fx.Int32(128),
            b_base_tile + self._q * fx.Int32(128),
        )
        return _uniform_i32(base_tile // fx.Int32(32))

    def set_tile_voff(self, G):
        """This tile's per-lane gather offset with the K-block folded out (it rides
        the uniform soffset instead): computed once per tile, so the K loop carries
        no per-gather voffset math."""
        grp = fx.Int32(G) + self._blk
        self._voff_g = fx.as_ir_value(
            (grp * fx.Int32(self.row_i32) + self._in16 * fx.Int32(4)) * fx.Int32(4)
        )

class ScaleLoaderLDS:
    """The wave's scale dwords of a block: lane (klane, r) reads the dword of group
    (half, gi) at its (klane, mlane) position -- the layout the MFMA wants."""

    def __init__(self, n_tiles, lane_id, quarter, lds_base_ptr, region_off):
        self.n_groups = (n_tiles + 1) // _PACK
        self.n_tiles = n_tiles
        self.quarter = quarter
        self.lane_id = lane_id
        self._region_base = (
            fx.Int32(fx.ptrtoint(lds_base_ptr))
            + fx.Int32(region_off)
        )

    def read_half(self, slot, half):
        L = self.lane_id
        base = (
            self._region_base
            + fx.Int32(slot) * fx.Int32(_SCALE_SLOT_BYTES)
            + fx.Int32((L // 4) * 16 + (L % 4) * 4)
        )
        out = []
        for gi in range_constexpr(self.n_groups):
            row = self.quarter * (16 * self.n_tiles) + half * (32 * self.n_tiles) + gi * 32
            group = row // 32
            ptr = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(base + group * 256))
            lo = fx.Int32(_llvm.LoadOp(fx.Int32.ir_type, ptr, alignment=4).result)
            if const_expr(self.n_tiles % 2 == 1):
                ptr2 = _llvm.inttoptr(_lds_ptr_t(), fx.as_ir_value(base + (group + 1) * 256))
                hi = fx.Int32(_llvm.LoadOp(fx.Int32.ir_type, ptr2, alignment=4).result)
                odd = (row % 32) != 0
                packed = (lo.shrui(fx.Int32(8)) & 0x00ff00ff) | ((hi & 0x00ff00ff) << 8)
                lo = odd.select(packed, lo)
            out.append(lo)

        return out

    def read(self, slot):
        return self.read_half(slot, 0), self.read_half(slot, 1)

class StoreCLds(StoreCLdsBase):
    """``StoreCLds`` for any quadrant width: 16-byte pieces of a 16-row block, lane-linear
    over ceil(16 * 2 n_b / 64) reads; lanes past the block repeat a piece (the same bytes
    to the same address)."""

    def __init__(self, C, c_rows, c_cols, c_idx_fn, lds_base_ptr, wave_id, n_a, n_b, nt=False):
        self.c_cols = c_cols
        self.c_rows = c_rows
        self.c_idx_fn = c_idx_fn
        self.n_a = n_a
        self.n_b = n_b
        self.pieces = 2 * n_b
        per_block = 16 * self.pieces
        self.slabs = (per_block + 63) // 64
        self.buf_bytes = 16 * 32 * n_b
        self.cpol = 2 if nt else 0
        self.rsrc = buffer_ops.create_buffer_resource(
            C, max_size=False, num_records_bytes=c_rows * c_cols * 2
        )
        lane = fx.thread_idx.x % 64
        g = lane // 16
        r = lane % 16
        row_b = 32 * n_b
        base = fx.Int32(fx.ptrtoint(lds_base_ptr)) + wave_id * fx.Int32(2 * self.buf_bytes)
        self.w_off = [base + r * fx.Int32(row_b) + fx.Int32(32 * tj) + g * fx.Int32(8)
                      for tj in range_constexpr(n_b)]
        self.r_off, self.st_offs = [], []
        for h in range_constexpr(self.slabs):
            left = per_block - 64 * h
            lin = fx.Int32(64 * h) + (lane if left >= 64 else lane % left)
            row, q = lin // self.pieces, lin % self.pieces
            self.r_off.append(base + row * fx.Int32(row_b) + q * fx.Int32(16))
            self.st_offs.append((row * c_cols) * 2 + q * 16)

    def _read_store(self, base_row, base_col, ti, h, buf):
        ptr = _llvm.inttoptr(
            _lds_ptr_t(), fx.as_ir_value(self.r_off[h] + fx.Int32(buf * self.buf_bytes))
        )
        vec4 = _ir.VectorType.get([4], fx.Int32.ir_type)
        data = Vec(_llvm.LoadOp(vec4, ptr, alignment=16).result)
        off = ((base_row + ti * 16) * self.c_cols + base_col) * 2 + self.st_offs[h]
        buffer_ops.buffer_store(
            data, self.rsrc, off, offset_is_bytes=True, cache_modifier=self.cpol
        )


def compile_mxfp8_gemm_n4992_k6144(tiles=(4, 2), store_nt=False, grid_max=256):
    """The persistent kernel with ``tiles`` = (n_a, n_b) 16-row / 16-column MFMA tiles per
    wave quadrant; ``grid_max`` above the work-item count launches a plain grid."""
    K, N = 6144, 4992
    N_TILES_A, N_TILES_B = tiles
    assert N_TILES_A in (2, 3, 4) and N_TILES_B in (2, 3)
    BLOCK_M = 64 * N_TILES_A
    BLOCK_N = 64 * N_TILES_B
    assert N % BLOCK_N == 0
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2
    assert K % 256 == 0
    K_ITERS = K // BLOCK_K  # even
    UNROLL = 4 if (K_ITERS - 4) % 4 == 0 else 2
    assert (K_ITERS - 4) % UNROLL == 0, K_ITERS
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert (
        LDS_BLOCK_M * BLOCK_K <= _LDS_BUF_BYTES
        and LDS_BLOCK_N * BLOCK_K <= _LDS_BUF_BYTES
    )
    # grid_max: workgroups of the launch (256 = one per CU, persistent); a larger value gives
    # every work item its own workgroup (a plain grid), which no co-resident kernel holds back
    GRID_MAX = grid_max
    name = (f"flydsl_mxfp8_n4992_k6144_{BLOCK_M}x{BLOCK_N}" + ("_nt" if store_nt else "")
            + ("" if grid_max == 256 else f"_g{grid_max}"))
    SCALE_SLOTS = _SCALE_SLOTS
    OUT_LDS_BYTES = _N_WAVES * 2 * 16 * 32 * N_TILES_B

    @fx.struct
    class SharedStorage:
        all_lds: fx.Array[fx.Int8, 8 * _LDS_BUF_BYTES, 16]
        scale_lds: fx.Array[fx.Int8, SCALE_SLOTS * _SCALE_SLOT_BYTES, 16]
        out_lds: fx.Array[fx.Int8, OUT_LDS_BYTES, 16]

    @flyc.kernel(name=name)
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        n_work: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        _base_ptr = lds.all_lds.ptr

        def _buf(idx):
            return _Buf(_base_ptr, idx * _LDS_BUF_BYTES)

        a_cur0, a_cur1, a_next0, a_next1 = _buf(0), _buf(1), _buf(2), _buf(3)
        b_cur0, b_cur1, b_next0, b_next1 = _buf(4), _buf(5), _buf(6), _buf(7)

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        n_pid_m = _ceildiv(c_m, BLOCK_M)
        n_blocks = c_n // BLOCK_N
        grid = _min(fx.Int32(n_work), fx.Int32(GRID_MAX))
        wg = fx.Int32(fx.block_idx.x)
        my_tiles = (fx.Int32(n_work) - wg + grid - fx.Int32(1)) // grid
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A_K_STEP = BLOCK_K
        B_K_STEP = 2 * 1024  # preshuffled (16, 16): a 16-row block spans 2 KB per 128 K

        def tile_coords(t):
            """work item -> (tile_i, tile_j) in the XCD order of ``_xcd_swizzle``; a negative /
            out-of-range index lands beyond M so its stores are dropped by the buffer bounds."""
            return _xcd_swizzle(n_pid_m, n_blocks, t)

        def tile_bases(ti, tj):
            """the g2s bases of a tile (row-major A, preshuffled W) and its scale groups"""
            return (
                (ti * BLOCK_M) * K,
                (ti * BLOCK_M + LDS_BLOCK_M) * K,
                (tj * BLOCK_N) * K,
                (tj * BLOCK_N + LDS_BLOCK_N) * K,
            )

        mfma = MfmaRowStationary(N_TILES_A, N_TILES_B)
        _scale_base_ptr = lds.scale_lds.ptr
        scale_gather = ScaleGatherLDS(
            A_scale,
            B_scale,
            K,
            lane_id,
            wave_id,
            _scale_base_ptr,
            c_m,
            c_n,
            N_TILES_A,
            N_TILES_B,
        )
        scale_gather.set_wave_base()
        # the two scale-slot LDS addresses (slot 0 / 1) precomputed once; the
        # gather's K-block offset rides the uniform soffset inside the K loop
        _gm0_s = (
            scale_gather._wave_base_s,
            fx.as_ir_value(
                fx.Int32(scale_gather._wave_base_s) + fx.Int32(_SCALE_SLOT_BYTES)
            ),
        )
        a_scale_ld = ScaleLoaderLDS(
            N_TILES_A, lane_id, wave_i, _scale_base_ptr, _SCALE_A_REGION
        )
        b_scale_ld = ScaleLoaderLDS(
            N_TILES_B, lane_id, wave_j, _scale_base_ptr, _SCALE_B_REGION
        )

        def _slot(b):
            return fx.Int32(b) % fx.Int32(SCALE_SLOTS)

        gl_off_a = _global_swizzle(lane_id, wave_id, K, N_TILES_A, False)
        gl_off_b = _global_swizzle(lane_id, wave_id, K, N_TILES_B, True)
        a_rsrc = buffer_ops.create_buffer_resource(
            A, max_size=False, num_records_bytes=c_m * K
        )
        b_rsrc = buffer_ops.create_buffer_resource(
            B_T, max_size=False, num_records_bytes=c_n * K
        )
        a_g2s = G2SLoaderAsm(a_rsrc, gl_off_a, N_TILES_A, wave_id)
        b_g2s = G2SLoaderAsm(b_rsrc, gl_off_b, N_TILES_B, wave_id)
        a_g2s.set_wave_base(_base_ptr)
        b_g2s.set_wave_base(_base_ptr)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        store_c = (StoreCLds if N_TILES_B % 2 else StoreCLdsBase)(
            C,
            c_m,
            c_n,
            mfma.idx,
            lds.out_lds.ptr,
            wave_id,
            N_TILES_A,
            N_TILES_B,
            nt=store_nt,
        )

        # a step issues [ac0 (n_a) | bc0 (n_b) | bc1 (n_b) | ac1 (n_a)] loads per wave: at a
        # step's top the previous step's may all be in flight; mid-step the previous step's
        # ac1 and this step's first two batches
        _MAIN_VMCNT = 2 * N_TILES_A + 2 * N_TILES_B
        _SEG2_VMCNT = 2 * N_TILES_A + N_TILES_B

        def _read_scale_thunks(block, holder):
            # the caller passes the LDS slot itself (the block's K offset rides the
            # uniform soffset, so no block -> slot math is left)
            s = block

            def _r(dst, ld, half, _s=s):
                holder[dst] = ld.read_half(_s, half)

            return [
                lambda: _r(0, a_scale_ld, 0),
                lambda: _r(1, a_scale_ld, 1),
                lambda: _r(2, b_scale_ld, 0),
                lambda: _r(3, b_scale_ld, 1),
            ]

        def _g2s_bases(bases, step):
            a0, a1, b0, b1 = bases
            st = fx.Int32(step)
            return (
                fx.Int32(a0) + st * fx.Int32(A_K_STEP),
                fx.Int32(a1) + st * fx.Int32(A_K_STEP),
                fx.Int32(b0) + st * fx.Int32(B_K_STEP),
                fx.Int32(b1) + st * fx.Int32(B_K_STEP),
            )

        def _one_step(
            parity,
            a0f,
            b0f,
            b1f_in,
            sc,
            accs,
            bufs,
            load,
            gather,
            read_block,
            zero_acc=False,
            extra=None,
        ):
            """One K-step: this step's MFMAs with, interleaved, the LDS reads for the next
            step, the global loads ``load`` of the step after (the resolved (a0, a1, b0,
            b1) soffsets), at even steps the gather ``gather`` of the next scale block (the
            precomputed (m0, soff)), at odd steps the read of scale block ``read_block``
            (the LDS slot itself) into the VGPR carry;
            ``extra(g, c00f, c01f)`` (called before group g = 0..3 with the accumulators
            finished so far, None before) returns more thunks for that group: the stores
            of this tile's c00 / c01 in the last step, of the previous tile's c10 / c11
            in the first."""
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            saR0, saR1, sbC0, sbC1 = sc
            c00f, c01f, c10f, c11f = accs
            _a1 = [None] * N_TILES_A
            _a0n = [None] * N_TILES_A
            _b0n = [None] * N_TILES_B
            _b1n = [None] * N_TILES_B
            a0_off, a1_off, b0_off, b1_off = load
            _scn = [None, None, None, None]
            _rd_scn = _read_scale_thunks(read_block, _scn) if parity == 1 else []
            _sc_gather = (
                []
                if parity == 1
                else [lambda: scale_gather.gather_soff(gather[0], gather[1])]
            )
            ex = extra if extra is not None else (lambda g, c00, c01: ())

            wait_barrier(_MAIN_VMCNT)
            il = _riffle(
                _g2s_thunks(a_g2s, ac0, a0_off, N_TILES_A),
                _s2r_thunks(a_s2r, ac1, _a1, N_TILES_A, False),
            ) + list(ex(0, None, None))
            c00f = mfma.call(
                a0f, b0f, c00f, saR0, sbC0, parity, interleave=il, zero_acc=zero_acc
            )
            il = _riffle(
                _g2s_thunks(b_g2s, bc0, b0_off, N_TILES_B), list(ex(1, c00f, None))
            )
            c01f = mfma.call(
                a0f, b1f_in, c01f, saR0, sbC1, parity, interleave=il, zero_acc=zero_acc
            )
            a1f = _a1

            wait_barrier(_SEG2_VMCNT)
            il = (_riffle(_g2s_thunks(b_g2s, bc1, b1_off, N_TILES_B), _s2r_thunks(a_s2r, an0, _a0n, N_TILES_A, False))
                  + _sc_gather + _rd_scn[:2] + list(ex(2, c00f, c01f)))  # fmt: skip
            c10f = mfma.call(
                a1f, b0f, c10f, saR1, sbC0, parity, interleave=il, zero_acc=zero_acc
            )
            a0nf = _a0n
            il = (
                _riffle(
                    _g2s_thunks(a_g2s, ac1, a1_off, N_TILES_A),
                    _s2r_thunks(b_s2r, bn0, _b0n, N_TILES_B, True)
                    + _s2r_thunks(b_s2r, bn1, _b1n, N_TILES_B, True),
                )
                + _rd_scn[2:]
                + list(ex(3, c00f, c01f))
            )
            c11f = mfma.call(
                a1f, b1f_in, c11f, saR1, sbC1, parity, interleave=il, zero_acc=zero_acc
            )
            b0nf, b1nf = _b0n, _b1n

            sc_next = (_scn[0], _scn[1], _scn[2], _scn[3]) if parity == 1 else sc
            new_bufs = (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)
            return a0nf, b0nf, b1nf, sc_next, (c00f, c01f, c10f, c11f), new_bufs

        bufs0 = (a_cur0, a_cur1, a_next0, a_next1, b_cur0, b_cur1, b_next0, b_next1)

        def _swap_bufs(bufs):
            ac0, ac1, an0, an1, bc0, bc1, bn0, bn1 = bufs
            return (an0, an1, ac0, ac1, bn0, bn1, bc0, bc1)

        n_a = N_TILES_A
        n_b = N_TILES_B
        n_ga = (N_TILES_A + 1) // _PACK
        n_gb = (N_TILES_B + 1) // _PACK
        n_sc = 2 * n_ga + 2 * n_gb
        _R = fx.as_ir_value

        def _flat_sc(sc):
            saR0, saR1, sbC0, sbC1 = sc
            return (
                [_R(v) for v in saR0]
                + [_R(v) for v in saR1]
                + [_R(v) for v in sbC0]
                + [_R(v) for v in sbC1]
            )

        def _unflat_sc(flat):
            o = 0
            saR0 = list(flat[o : o + n_ga])
            o += n_ga
            saR1 = list(flat[o : o + n_ga])
            o += n_ga
            sbC0 = list(flat[o : o + n_gb])
            o += n_gb
            sbC1 = list(flat[o : o + n_gb])
            return (saR0, saR1, sbC0, sbC1)

        # ---- prologue: the first tile's steps 0 / 1 and its scale blocks 0 / 1 ----
        t0 = wg
        ti0, tj0 = tile_coords(t0)
        bases0 = tile_bases(ti0, tj0)
        G0 = scale_gather.tile_group(ti0 * BLOCK_M, tj0 * BLOCK_N)
        scale_gather.set_tile_voff(G0)
        scale_gather.gather_soff(_gm0_s[0], 0)
        scale_gather.gather_soff(_gm0_s[1], 256)
        s0 = _g2s_bases(bases0, 0)
        s1 = _g2s_bases(bases0, 1)
        a_g2s.load(a_cur0, s0[0])
        b_g2s.load(b_cur0, s0[2])
        b_g2s.load(b_cur1, s0[3])
        a_g2s.load(a_cur1, s0[1])
        a_g2s.load(a_next0, s1[0])
        b_g2s.load(b_next0, s1[2])
        b_g2s.load(b_next1, s1[3])
        a_g2s.load(a_next1, s1[1])
        # 2 gathers + 32 loads in flight: a_cur0 landed at vmcnt(28), b_cur0 / b_cur1 at 20
        wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier((3 * N_TILES_A) + (2 * N_TILES_B))
        b0_frag = b_s2r.load(b_cur0, preshuffled=True)
        b1_frag = b_s2r.load(b_cur1, preshuffled=True)
        sc0_saR0, sc0_saR1 = a_scale_ld.read(_slot(0))
        sc0_sbC0, sc0_sbC1 = b_scale_ld.read(_slot(0))
        sc0 = (sc0_saR0, sc0_saR1, sc0_sbC0, sc0_sbC1)
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        c10_prev = [zero4] * N_ACCUMS
        c11_prev = [zero4] * N_ACCUMS

        # ---- the tile loop: state = fragments of the tile's step 0, its scale block 0, the
        # previous tile's c10 / c11 (stored during this tile's first step), the tile index
        tile_state = (
            [_R(x) for x in a0_frag] + [_R(x) for x in b0_frag] + [_R(x) for x in b1_frag] + _flat_sc(sc0)
            + [_R(x) for x in c10_prev] + [_R(x) for x in c11_prev] + [_R(fx.Int32(t0)), _R(fx.Int32(-1))]
        )  # fmt: skip
        for jt, tstate in range(0, my_tiles, 1, init=tile_state):
            off = 0
            a0f = list(tstate[off : off + n_a])
            off += n_a
            b0f = list(tstate[off : off + n_b])
            off += n_b
            b1f = list(tstate[off : off + n_b])
            off += n_b
            sc = _unflat_sc(tstate[off : off + n_sc])
            off += n_sc
            c10p = list(tstate[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11p = list(tstate[off : off + N_ACCUMS])
            off += N_ACCUMS
            t_cur = fx.Int32(tstate[off])
            t_prev = fx.Int32(tstate[off + 1])
            t_next = t_cur + grid
            ti, tj = tile_coords(t_cur)
            bases = tile_bases(ti, tj)
            G = scale_gather.tile_group(ti * BLOCK_M, tj * BLOCK_N)
            scale_gather.set_tile_voff(G)
            # the next tile (the last tile loads a tile beyond the grid: rows beyond M read
            # zero through the buffer bounds and nothing of it is stored)
            tin, tjn = tile_coords(t_next)
            bases_n = tile_bases(tin, tjn)
            Gn = scale_gather.tile_group(tin * BLOCK_M, tjn * BLOCK_N)
            # the previous tile's c10 / c11 store bases (row beyond M when there is none)
            tip, tjp = tile_coords(t_prev)
            has_prev = t_prev >= fx.Int32(0)
            prev_row1 = fx.arith.select(
                has_prev,
                tip * BLOCK_M + wave_i * (N_TILES_A * 16) + LDS_BLOCK_M,
                fx.Int32(c_m),
            )
            prev_col0 = fx.arith.select(
                has_prev, tjp * BLOCK_N + wave_j * (N_TILES_B * 16), fx.Int32(0)
            )
            prev_col1 = fx.arith.select(has_prev, prev_col0 + LDS_BLOCK_N, fx.Int32(0))
            base_row = ti * BLOCK_M + wave_i * (N_TILES_A * 16)
            base_col = tj * BLOCK_N + wave_j * (N_TILES_B * 16)

            accs = (
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
                [None] * N_ACCUMS,
            )
            # step 0: zero accumulators, the previous tile's c10 / c11 stores spread over the
            # groups before the new c10 (born in group 2) / c11 (group 3), gather block 1
            prev_thunks = store_c.thunks(c10p, prev_row1, prev_col0) + store_c.thunks(
                c11p, prev_row1, prev_col1
            )
            n_prev3 = (
                len(prev_thunks) + 2
            ) // 3  # over the three groups before the new c11
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                0, a0f, b0f, b1f, sc, accs, bufs0, _g2s_bases(bases, 2), (_gm0_s[1], 256), None, zero_acc=True,
                extra=lambda g, c00, c01: prev_thunks[n_prev3 * g : n_prev3 * (g + 1)],
            )  # fmt: skip
            # step 1: read block 1
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                1, a0f, b0f, b1f, sc, accs, bufs, _g2s_bases(bases, 3), None, 1
            )
            # the four g2s streams advance by uniform strides and the gather
            # block by 256 B; carry them as uniform accumulators (incremented
            # per step) instead of recomputing base | (step << s) every K-step
            ka0 = fx.Int32(bases[0]) + fx.Int32(4 * A_K_STEP)
            ka1 = fx.Int32(bases[1]) + fx.Int32(4 * A_K_STEP)
            kb0 = fx.Int32(bases[2]) + fx.Int32(4 * B_K_STEP)
            kb1 = fx.Int32(bases[3]) + fx.Int32(4 * B_K_STEP)
            kg = fx.Int32(512)  # gather block soff at kk=2: block 2 << 8
            init_state = (
                [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                + [_R(x) for x in accs[0]] + [_R(x) for x in accs[1]] + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]]
                + [_R(ka0), _R(ka1), _R(kb0), _R(kb1), _R(kg)]
            )  # fmt: skip
            for kk, state in range(2, K_ITERS - 2, UNROLL, init=init_state):
                off = 0
                a0f = list(state[off : off + n_a])
                off += n_a
                b0f = list(state[off : off + n_b])
                off += n_b
                b1f = list(state[off : off + n_b])
                off += n_b
                sc = _unflat_sc(state[off : off + n_sc])
                off += n_sc
                c00f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c01f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c10f = list(state[off : off + N_ACCUMS])
                off += N_ACCUMS
                c11f = list(state[off : off + N_ACCUMS])
                accs = (c00f, c01f, c10f, c11f)
                bufs = bufs0
                off += N_ACCUMS  # past c11f: the uniform address accumulators
                ka0 = fx.Int32(state[off])
                ka1 = fx.Int32(state[off + 1])
                kb0 = fx.Int32(state[off + 2])
                kb1 = fx.Int32(state[off + 3])
                kg = fx.Int32(state[off + 4])
                for u in range_constexpr(UNROLL):
                    if const_expr(UNROLL == 4):
                        # kk == 2 (mod 4): u=0 gathers an even block (slot 0), u=2 an
                        # odd one (slot 1) -- gather m0 slots folded; read slots stay
                        # runtime (see below).
                        lds_off = (ka0, ka1, kb0, kb1)
                        gath = None
                        if const_expr(u % 2 == 0):
                            gath = (
                                _gm0_s[u // 2],
                                kg if u == 0 else kg + fx.Int32(256),
                            )
                        rslot = None
                        if const_expr(u % 2 == 1):
                            # keep the read slot a runtime value: folding it to a
                            # constant made LLVM un-pair the scale ds_read2st64
                            # into 2 ds_read_b32 (+2 LDS/step at the ISA gate)
                            rslot = (
                                fx.Int32(kk) // fx.Int32(2) + fx.Int32(1 + u // 2)
                            ) % fx.Int32(2)
                    else:
                        kc = fx.Int32(kk) + fx.Int32(u)
                        block = kc // fx.Int32(2) + fx.Int32(1)
                        lds_off = _g2s_bases(bases, kc + fx.Int32(2))
                        gath = None
                        if const_expr(u % 2 == 0):
                            gath = (
                                fx.as_ir_value(
                                    fx.Int32(scale_gather._wave_base_s)
                                    + (block % fx.Int32(2))
                                    * fx.Int32(_SCALE_SLOT_BYTES)
                                ),
                                block * fx.Int32(256),
                            )
                        rslot = None
                        if const_expr(u % 2 == 1):
                            rslot = block % fx.Int32(2)
                    a0f, b0f, b1f, sc, accs, bufs = _one_step(
                        u % 2,
                        a0f,
                        b0f,
                        b1f,
                        sc,
                        accs,
                        bufs,
                        lds_off,
                        gath,
                        rslot,
                    )
                    ka0 = ka0 + fx.Int32(A_K_STEP)
                    ka1 = ka1 + fx.Int32(A_K_STEP)
                    kb0 = kb0 + fx.Int32(B_K_STEP)
                    kb1 = kb1 + fx.Int32(B_K_STEP)
                new_state = (
                    [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                    + [_R(x) for x in accs[0]] + [_R(x) for x in accs[1]] + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]]
                    + [_R(ka0), _R(ka1), _R(kb0), _R(kb1), _R(kg + fx.Int32(UNROLL * 128))]
                )  # fmt: skip
                state = yield new_state

            off = 0
            a0f = list(state[off : off + n_a])
            off += n_a
            b0f = list(state[off : off + n_b])
            off += n_b
            b1f = list(state[off : off + n_b])
            off += n_b
            sc = _unflat_sc(state[off : off + n_sc])
            off += n_sc
            c00f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c01f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c10f = list(state[off : off + N_ACCUMS])
            off += N_ACCUMS
            c11f = list(state[off : off + N_ACCUMS])
            accs = (c00f, c01f, c10f, c11f)
            # step K_ITERS - 2 (even): loads of the next tile's step 0, gather of its block 0
            scale_gather.set_tile_voff(Gn)
            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                0,
                a0f,
                b0f,
                b1f,
                sc,
                accs,
                bufs0,
                _g2s_bases(bases_n, 0),
                (_gm0_s[0], 0),
                None,
            )
            # step K_ITERS - 1 (odd): loads of the next tile's step 1, read of its block 0;
            # this tile's c00 / c01 stored over the groups after their last MFMAs (c00: 1..3,
            # c01: 2..3; the two quadrants' thunks stay concatenated, they share the LDS buffer)
            n_q3 = (
                2 * store_c.n_thunks + 2
            ) // 3  # thunks per group: two quadrants over 3 groups

            def _last_extra(g, c00, c01):
                # (compile-time python only: no mutable state -- the DSL rewriter turns
                # body-level literals into None -- and every `if` under const_expr)
                if const_expr(g == 0):
                    return ()
                t00 = store_c.thunks(c00, base_row, base_col)
                if const_expr(g == 1):
                    return t00[:n_q3]
                t01 = store_c.thunks(c01, base_row, base_col + LDS_BLOCK_N)
                if const_expr(g == 2):
                    return t00[n_q3:] + t01[: 2 * n_q3 - len(t00)]
                return t01[2 * n_q3 - len(t00) :]

            a0f, b0f, b1f, sc, accs, bufs = _one_step(
                1,
                a0f,
                b0f,
                b1f,
                sc,
                accs,
                bufs,
                _g2s_bases(bases_n, 1),
                None,
                0,
                extra=_last_extra,
            )
            new_tile_state = (
                [_R(x) for x in a0f] + [_R(x) for x in b0f] + [_R(x) for x in b1f] + _flat_sc(sc)
                + [_R(x) for x in accs[2]] + [_R(x) for x in accs[3]] + [_R(t_next), _R(t_cur)]
            )  # fmt: skip
            tstate = yield new_tile_state

        # ---- epilogue: the last tile's c10 / c11 ----
        off = n_a + 2 * n_b + n_sc
        c10_last = list(tstate[off : off + N_ACCUMS])
        off += N_ACCUMS
        c11_last = list(tstate[off : off + N_ACCUMS])
        off += N_ACCUMS
        t_last = fx.Int32(tstate[off + 1])
        til, tjl = tile_coords(t_last)
        last_row1 = til * BLOCK_M + wave_i * (N_TILES_A * 16) + LDS_BLOCK_M
        last_col0 = tjl * BLOCK_N + wave_j * (N_TILES_B * 16)
        store_c.store(c10_last, last_row1, last_col0)
        store_c.store(c11_last, last_row1, last_col0 + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm_n4992_k6144(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        n_work: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = _min(n_work, fx.Int32(GRID_MAX))
        kernel_gemm(
            A, B_T, C, A_scale, B_scale, c_m, c_n, n_work,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)  # fmt: skip

    launch_gemm_n4992_k6144.kernel_name = name
    return launch_gemm_n4992_k6144
