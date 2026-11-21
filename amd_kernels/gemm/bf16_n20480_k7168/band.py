# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# Adapted from AMD-AGI/AKLR BF16 decode kernels (MIT), revision 551613c81ea1f232cadb72e6f4647f96152e67d4, through
# aiter `ops/flydsl/kernels/gemm_a16w16_band_gfx950.py` (build_band_stream).
"""Band-stream bf16 GEMM, N 20480 x K 7168, 1-96 rows: the lm_head shard with a row map and a push epilogue.

Each of 256 workgroups owns 64 line-aligned weight rows plus a 16-row strip grouped by XCD (80 vocabulary columns) and
all rows of the activation (one M band, BM <= 96). A and B stream HBM / L2 -> LDS by staged `buffer_load ... lds`
DMAs into a ring of STAGES slots (BLOCK_K = 128); fragment reads and the next stage's DMAs are interleaved with the
MFMAs. B is non-temporal; A stays cached; each workgroup starts its K loop at a staggered block (K_STAGGER) so the
concurrent reads of A spread over L2 channels; each output keeps one fixed K summation order, no split-K, no atomics.

Differences from the aiter kernel: the row count is a runtime argument (rows past it read zero through the buffer
bound), the activation rows are read through an optional device row map, and
the push modes send the output rows straight to the receivers' buffers of the exchange
(`collectives/all_to_all_bf16_n20480_ranks8`: the owner of each row, or every rank) with system-scope stores, then
publish with a two-level ticket (per XCD, then across the eight XCDs).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_d
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector

from .flags import SYS_CM, flag_add
from .reference import K, N

# (max rows, BM, BN, stages, M waves, N waves, K stagger, output cache policy, BLOCK_K): the bands of the lm_head shard
BANDS = (
    (16, 16, 96, 5, 1, 2, 3, 0, 128),
    (32, 32, 96, 5, 2, 2, 3, 0, 128),
    (48, 48, 96, 3, 1, 3, 1, 0, 128),
    (64, 64, 96, 3, 2, 2, 1, 2, 128),
    (80, 80, 80, 3, 1, 5, 1, 17, 128),
    (96, 96, 96, 3, 2, 2, 1, 2, 128),
)
MAX_ROWS = BANDS[-1][0]
_OOB = 0x7FFFFFF0  # a byte offset past every buffer bound: the DMA reads zeros with no traffic
_OOB_ELEMS = 0x3FFFFFF0  # the same in bf16 elements: a store there is dropped
V = 8 * N  # vocabulary row of the gathered logits
RECV_ROWS = 256  # rows of a destination buffer the stores address (the exchange's capacity bound)
FLAGS = 16  # index of the first flag block in the exchange's peers table (two buffers of eight bases first)


def _as_i32(v):
    v = fx.arith.unwrap(v)
    t = v.type
    if isinstance(t, ir.IntegerType):
        if t.width == 32:
            return v
        if t.width > 32:
            return std_arith.TruncIOp(T.i32, v).result
        return std_arith.ExtSIOp(T.i32, v).result
    return std_arith.IndexCastOp(T.i32, v).result


def _bstore(data, rsrc, elem_off, aux):
    data = fx.arith.unwrap(data)
    off = std_arith.MulIOp(_as_i32(elem_off), fx.arith.constant(data.type.element_type.width // 8)).result
    rocdl_d.RawPtrBufferStoreOp(data, rsrc, off, fx.arith.constant(0), aux=ir.IntegerAttr.get(T.i32, aux))


def _swizzle_xor16(row, col_in_bytes, k_blocks16):
    return col_in_bytes ^ ((row % k_blocks16) * 16)


def _lds_ptr3(base_i64, byte_off_i64):
    v = fx.arith.unwrap(base_i64 + byte_off_i64)
    return llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<3>"), v).result


def _gep(base_ptr, byte_offset):
    if isinstance(byte_offset, int):
        dyn, raw = [], [byte_offset]
    else:
        dyn, raw = [fx.arith.unwrap(byte_offset)], [-(2**31)]
    return llvm.GEPOp(base_ptr.type, base_ptr, dyn, raw, T.i8, None).result


def band_of(m: int) -> tuple:
    return next(row for row in BANDS if m <= row[0])


def compile_band_gemm(band: tuple, push: str | None = None, gather: bool = False):
    """The launcher of one M band; ``gather``: the activation rows come through a device row map; ``push``: the
    row-mapped GEMM sending its rows to ``owner`` ranks or ``all`` (implies the row map)."""
    _, TILE_M, TILE_N, STAGES, BLOCK_M_WARPS, BLOCK_N_WARPS, K_STAGGER, C_AUX, BLOCK_K = band
    WARP_SIZE = 64
    VMCNT_MAX = 63
    DTYPE_BYTES = 2
    LDG_VEC_SIZE = 8
    DMA_BYTES = 16
    WMMA_M = 16
    WMMA_N = 16
    WMMA_K = 32
    WMMA_FRAG_VALUES = 8
    WMMA_C_FRAG_VALUES = 4
    k, n = K, N
    BLOCK_K_LOOPS = k // BLOCK_K
    assert k % BLOCK_K == 0 and BLOCK_K_LOOPS >= STAGES
    WARP_K_STEPS = BLOCK_K // WMMA_K
    BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE
    NUM_WAVES = BLOCK_M_WARPS * BLOCK_N_WARPS
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WMMA_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WMMA_N
    assert WARP_M_STEPS >= 1 and WARP_N_STEPS >= 1
    WARP_M = WARP_M_STEPS * WMMA_M
    WARP_N = WARP_N_STEPS * WMMA_N
    BLOCK_M = BLOCK_M_WARPS * WARP_M
    BLOCK_N = BLOCK_N_WARPS * WARP_N
    N_REAL = n // 256
    N_BLOCKS = 256
    STRIP = N_REAL - 64
    assert BLOCK_M == TILE_M and BLOCK_N == TILE_N
    assert N_BLOCKS * N_REAL == n and N_BLOCKS % 8 == 0 and N_REAL <= BLOCK_N
    assert STRIP > 0 and STRIP % LDG_VEC_SIZE == 0
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES
    AS_ELEMS = STAGES * BLOCK_M * BLOCK_K
    # C is staged in LDS at a row stride padded by one store vector, keeping vector reads aligned.
    C_STRIDE = BLOCK_N + LDG_VEC_SIZE
    CMN_ELEMS = BLOCK_M * C_STRIDE
    assert CMN_ELEMS <= AS_ELEMS
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    ROWS_PER_LOAD = BLOCK_THREADS // LDG_X_THREADS_AS
    # A B stage holds only the rows the DMAs write; MFMA rows past them feed unstored columns.
    LDG_REG_B_COUNT_AS = (N_REAL + ROWS_PER_LOAD - 1) // ROWS_PER_LOAD
    B_STAGE_ROWS = LDG_REG_B_COUNT_AS * ROWS_PER_LOAD
    BS_ELEMS = STAGES * B_STAGE_ROWS * BLOCK_K
    B_PAD_ELEMS = max(0, BLOCK_N - B_STAGE_ROWS) * BLOCK_K
    LDS_ELEMS = AS_ELEMS + BS_ELEMS + B_PAD_ELEMS
    B_LAST_PARTIAL = B_STAGE_ROWS > N_REAL
    A_CHUNKS = BLOCK_M * BLOCK_K // LDG_ASYNC_VEC_SIZE
    assert A_CHUNKS % BLOCK_THREADS == 0
    A_LOADS = A_CHUNKS // BLOCK_THREADS
    LDG_WAIT_COUNT = LDG_REG_B_COUNT_AS + A_LOADS
    assert (STAGES - 2) * LDG_WAIT_COUNT < VMCNT_MAX
    LdsArr = fx.Array[fx.BFloat16, LDS_ELEMS, 16]
    PIPE_ITEMS = [("a", i) for i in range(A_LOADS)] + [("b", i) for i in range(LDG_REG_B_COUNT_AS)]
    # Half a stage's DMAs (A first) go in K step 0 and the rest in K step 1, overlapping the MFMAs.
    _nd = len(PIPE_ITEMS)
    PIPE_BOUNDS = [0, (_nd + 1) // 2, _nd] + [_nd] * (WARP_K_STEPS - 2)
    assert push in (None, "owner", "all")
    rowmap = gather or bool(push)
    name = (f"bf16_gemm_n20480_k7168_band_tm{TILE_M}_bk{BLOCK_K}" + (f"_push_{push}" if push else "")
            + ("_gather" if gather and not push else ""))

    @flyc.kernel(name=name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, arg_indices: fx.Tensor, i32_xrows: fx.Int32,
               i32_m: fx.Int32, counter: fx.Tensor, arg_flag: fx.Int64, peers: fx.Tensor, counts: fx.Tensor,
               i32_rank: fx.Int32, i32_buf: fx.Int32):
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))
        bf16x8_ty = fx.Vector.make_type(LDG_VEC_SIZE, fx.BFloat16)
        a_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_xrows * (k * DTYPE_BYTES))
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=n * k * DTYPE_BYTES)
        c_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_m * (n * DTYPE_BYTES))
        lds_base = fx.SharedAllocator().allocate(LdsArr).peek().ptr
        b_lds_ptr = lds_base + AS_ELEMS
        a_lds_i64 = fx.Int64(fx.ptrtoint(lds_base))
        b_lds_i64 = a_lds_i64 + fx.Int64(AS_ELEMS * DTYPE_BYTES)

        def as_load(stage, row, col):
            elem_off = fx.Int64(stage) * (BLOCK_M * BLOCK_K) + fx.Int64(row) * BLOCK_K + fx.Int64(col)
            return fx.ptr_load(lds_base + elem_off, result_type=bf16x8_ty)

        def bs_load(stage, row, col):
            elem_off = fx.Int64(stage) * (B_STAGE_ROWS * BLOCK_K) + fx.Int64(row) * BLOCK_K + fx.Int64(col)
            return fx.ptr_load(b_lds_ptr + elem_off, result_type=bf16x8_ty)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        pid = fx.block_idx.x
        n_main = fx.Index(pid * 64)
        strip_idx = (pid % 8) * (N_BLOCKS // 8) + pid // 8
        n_strip = fx.Index(strip_idx * STRIP + (N_BLOCKS * 64 - 64))
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)
        warp_m_idx = wid // BLOCK_N_WARPS * WARP_M
        warp_n_idx = wid % BLOCK_N_WARPS * WARP_N
        ldmatrix_m_idx = w_tid % WMMA_M
        ldmatrix_k_vec_idx = w_tid // WMMA_M * WMMA_FRAG_VALUES
        c_frags = [acc_init] * (WARP_M_STEPS * WARP_N_STEPS)

        def mfma(a_frag, b_frag, c_frag):
            return rocdl.mfma_f32_16x16x32_bf16(T.vec(WMMA_C_FRAG_VALUES, T.f32), [b_frag, a_frag, c_frag, 0, 0, 0])

        def wait_barrier(vmcnt):
            llvm.InlineAsmOp(None, [], f"s_waitcnt vmcnt({vmcnt})\n\ts_barrier", "", has_side_effects=True)

        warp_offset = rocdl.readfirstlane(
            T.i64,
            arith.index_cast(T.i64, fx.Index(wid) * arith.constant(WARP_SIZE * DMA_BYTES, index=True)),
        )

        def buffer_load_lds_inline(rsrc, lds_ptr, global_offset, bits):
            llvm.InlineAsmOp(
                None,
                [lds_ptr, global_offset, rsrc],
                "s_mov_b32 m0, $0\n\tbuffer_load_dwordx4 $1, $2, 0 offen " + bits + " lds",
                "s,v,s",
                has_side_effects=True,
            )

        # the activation row of each A DMA of this thread, once: byte offset of its row start (the row map applied;
        # rows past the count read zeros through an offset past the bound) and its XOR-swizzled K column
        a_row_off, a_col_sw = [], []
        for ii in range_constexpr(A_LOADS):
            global_tid = BLOCK_THREADS * ii + tid
            m_local_idx = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            a_col_sw.append(_swizzle_xor16(m_local_idx, k_local_idx * DTYPE_BYTES, k_blocks16))
            live = fx.Int32(m_local_idx) < i32_m
            if const_expr(rowmap):
                src_row = fx.Int32(arg_indices[live.select(fx.Int32(m_local_idx), fx.Int32(0))])
            else:
                src_row = fx.Int32(m_local_idx)
            a_row_off.append(live.select(src_row * (k * DTYPE_BYTES), fx.Int32(_OOB)))

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_offset = _as_i32(a_row_off[ii] + fx.Int32(k_offset) * DTYPE_BYTES + a_col_sw[ii])
            if const_expr(lds_ptr is None):
                stage_off = arith.index_cast(T.i64, fx.Index(write_stage) * (BLOCK_M * BLOCK_K * DTYPE_BYTES))
                lds_ptr = _gep(_lds_ptr3(a_lds_i64, fx.Int64(stage_off)), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, bits="sc0")
            return lds_ptr

        def ldg_sts_a_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(A_LOADS):
                lds_ptr = ldg_sts_a_async_one(i, k_offset, lds_stage, lds_ptr)

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_tid = BLOCK_THREADS * ii + tid
            n_local_idx = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            col_in_bytes = k_local_idx * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(n_local_idx, col_in_bytes, k_blocks16)
            lo_row = ii * ROWS_PER_LOAD
            hi_row = lo_row + ROWS_PER_LOAD
            if const_expr(hi_row <= 64):
                row_idx = n_main + fx.Index(n_local_idx)
            elif const_expr(lo_row >= 64):
                row_idx = n_strip + fx.Index(n_local_idx)
            else:
                row_idx = fx.Index(
                    (fx.Int32(n_local_idx) < fx.Int32(64)).select(
                        fx.Int32(n_main) + fx.Int32(n_local_idx),
                        fx.Int32(n_strip) + fx.Int32(n_local_idx),
                    )
                )
            col_idx = fx.Index(k_offset + col_in_bytes // DTYPE_BYTES)
            global_offset = (row_idx * k + col_idx) * DTYPE_BYTES
            global_offset = arith.index_cast(T.i32, global_offset)
            if const_expr(B_LAST_PARTIAL and ii == LDG_REG_B_COUNT_AS - 1):
                # Rows past N_REAL belong to the next workgroup; out of bounds, they read zero.
                global_offset = fx.arith.unwrap(
                    (fx.Int32(n_local_idx) < fx.Int32(N_REAL)).select(fx.Int32(global_offset), fx.Int32(_OOB))
                )
            if const_expr(lds_ptr is None):
                stage_off = arith.index_cast(T.i64, fx.Index(write_stage) * (B_STAGE_ROWS * BLOCK_K * DTYPE_BYTES))
                lds_ptr = _gep(_lds_ptr3(b_lds_i64, fx.Int64(stage_off)), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            # B is read once; a non-temporal load keeps it from displacing the reused A lines.
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, bits="sc0 nt")
            return lds_ptr

        def ldg_sts_b_async(k_offset, lds_stage):
            lds_ptr = None
            for i in range_constexpr(LDG_REG_B_COUNT_AS):
                lds_ptr = ldg_sts_b_async_one(i, k_offset, lds_stage, lds_ptr)

        def load_b_frag(s, n_step, warp_atom_k_idx):
            row = warp_n_idx + n_step * WMMA_N + ldmatrix_m_idx
            col_in_bytes = (warp_atom_k_idx + ldmatrix_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(row, col_in_bytes, k_blocks16)
            return bs_load(s, row, col_in_bytes // DTYPE_BYTES)

        def load_a_frag(s, m_step, warp_atom_k_idx):
            row = warp_m_idx + m_step * WMMA_M + ldmatrix_m_idx
            col_in_bytes = (warp_atom_k_idx + ldmatrix_k_vec_idx) * DTYPE_BYTES
            col_in_bytes = _swizzle_xor16(row, col_in_bytes, k_blocks16)
            return as_load(s, row, col_in_bytes // DTYPE_BYTES)

        def frag_loads(s, kk):
            wk = kk * WMMA_K
            a_f = [load_a_frag(s, ii, wk) for ii in range_constexpr(WARP_M_STEPS)]
            b_f = [load_b_frag(s, ii, wk) for ii in range_constexpr(WARP_N_STEPS)]
            return a_f, b_f

        def pipe_compute(lds_stage, c_frags, dma=None):
            # Fragments are read one K step ahead.
            s = fx.Index(lds_stage)
            c_new = list(c_frags)
            ptrs = {"a": None, "b": None}
            cur = frag_loads(s, 0)
            for kk in range_constexpr(WARP_K_STEPS):
                if const_expr(dma is not None):
                    for j in range_constexpr(PIPE_BOUNDS[kk], PIPE_BOUNDS[kk + 1]):
                        kind, i = PIPE_ITEMS[j]
                        if const_expr(kind == "a"):
                            ptrs["a"] = ldg_sts_a_async_one(i, dma[0], dma[1], ptrs["a"])
                        else:
                            ptrs["b"] = ldg_sts_b_async_one(i, dma[0], dma[1], ptrs["b"])
                nxt = None
                if const_expr(kk + 1 < WARP_K_STEPS):
                    nxt = frag_loads(s, kk + 1)
                rocdl.sched_barrier(0)
                a_f, b_f = cur
                for ii in range_constexpr(WARP_M_STEPS):
                    for jj in range_constexpr(WARP_N_STEPS):
                        c_idx = ii * WARP_N_STEPS + jj
                        c_new[c_idx] = mfma(a_f[ii], b_f[jj], c_new[c_idx])
                rocdl.sched_barrier(0)
                cur = nxt
            return c_new

        def kwrap(off):
            return (off >= fx.Int32(k)).select(off - fx.Int32(k), off)

        # Staggered K starts spread concurrent A reads; each output keeps one fixed summation order.
        ks_begin = fx.Int32(pid * K_STAGGER % BLOCK_K_LOOPS * BLOCK_K)
        for s in range_constexpr(STAGES - 1):
            ks = kwrap(ks_begin + s * BLOCK_K)
            ldg_sts_a_async(ks, s)
            ldg_sts_b_async(ks, s)
        rocdl.sched_barrier(0)

        init_state = [ks_begin, arith.constant(0, index=True)] + c_frags
        for bki, state in range(0, BLOCK_K_LOOPS - (STAGES - 1), 1, init=init_state):
            k_offset = state[0]
            current_stage = fx.Index(state[1])
            c_frags = state[2:]
            next_stage = (current_stage + 1) % STAGES
            write_stage = (current_stage + STAGES - 1) % STAGES
            wait_barrier((STAGES - 2) * LDG_WAIT_COUNT)
            k_load = kwrap(k_offset + (STAGES - 1) * BLOCK_K)
            c_frags_new = pipe_compute(current_stage, c_frags, (k_load, write_stage))
            k_offset_next = kwrap(k_offset + fx.Int32(BLOCK_K))
            results = yield [k_offset_next, next_stage] + c_frags_new
        current_stage = fx.Index(results[1])
        c_frags = results[2:]
        for s in range_constexpr(0, STAGES - 1):
            wait_barrier((STAGES - 2 - s) * LDG_WAIT_COUNT)
            c_frags = pipe_compute(current_stage, c_frags)
            current_stage = (current_stage + 1) % STAGES

        c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        c_n_idx = w_tid % WMMA_N
        # C reuses the A ring; a barrier first when a tail step may still read it
        if const_expr(CMN_ELEMS > BLOCK_M * BLOCK_K or 0 in [(BLOCK_K_LOOPS - (STAGES - 1) + j) % STAGES
                                                            for j in range(STAGES - 1)]):
            fx.gpu.barrier()
        for ii in range_constexpr(WARP_M_STEPS):
            for jj in range_constexpr(WARP_N_STEPS):
                cast = Vector(c_frags[ii * WARP_N_STEPS + jj]).to(fx.BFloat16)
                lds_m = fx.Int64(warp_m_idx + ii * WMMA_M + c_n_idx)
                lds_n = fx.Int64(warp_n_idx + jj * WMMA_N + c_m_vec_idx)
                fx.ptr_store(cast, lds_base + (lds_m * C_STRIDE + lds_n))
        fx.gpu.barrier()

        if const_expr(push):
            dst = [buffer_ops.create_buffer_resource_from_addr(fx.Int64(peers[i32_buf * 8 + d]),
                                                           num_records_bytes=RECV_ROWS * V * 2)
                   for d in range_constexpr(8)]
            cum = [fx.Int32(0)]
            for d in range_constexpr(8):
                cum.append(cum[d] + fx.Int32(counts[d]))

        def store_c(m_local, n_local, base_col, aux):
            vec = fx.ptr_load(lds_base + (fx.Int64(m_local) * C_STRIDE + fx.Int64(n_local)), result_type=bf16x8_ty)
            if const_expr(push):
                # one 16-byte vector of output row r, columns base_col + n_local of this rank's 20480: to the row's
                # owner (counts: this rank's rows are sorted by owner) or to every rank (all-gather); rows past the
                # count and every other destination take an offset past the bound, a dropped store
                r = fx.Int32(m_local)
                live = r < i32_m
                if const_expr(push == "owner"):
                    owner = fx.Int32(0)
                    lo = fx.Int32(0)
                    for j in range_constexpr(1, 8):
                        ge = r >= cum[j]
                        owner = ge.select(fx.Int32(j), owner)
                        lo = ge.select(cum[j], lo)
                    off = (r - lo) * V + i32_rank * n + fx.Int32(base_col) + n_local
                else:
                    off = r * V + i32_rank * n + fx.Int32(base_col) + n_local
                for d in range_constexpr(8):
                    ok = (live & (owner == d)) if const_expr(push == "owner") else live
                    _bstore(vec, dst[d], ok.select(off, fx.Int32(_OOB_ELEMS)), SYS_CM)
            else:
                _bstore(vec, c_rsrc, m_local * n + fx.Int32(base_col) + n_local, aux)

        # rows past the count are out of the output buffer's bound: their stores are dropped
        for col0, width, base_col, aux in ((0, 64, n_main, C_AUX), (64, STRIP, n_strip, 0)):
            xt = width // LDG_VEC_SIZE
            nv = BLOCK_M * xt
            for i in range_constexpr((nv + BLOCK_THREADS - 1) // BLOCK_THREADS):
                global_tid = BLOCK_THREADS * i + tid
                m_local = global_tid // xt
                n_local = global_tid % xt * LDG_VEC_SIZE + col0
                if const_expr(nv % BLOCK_THREADS != 0 and i == nv // BLOCK_THREADS):
                    if fx.Int32(global_tid) < fx.Int32(nv):
                        store_c(m_local, n_local, base_col, aux)
                else:
                    store_c(m_local, n_local, base_col, aux)

        if const_expr(push):
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            # the same two-level ticket; the last arriver stores the sequence into this rank's word of every
            # receiver's flag block (byte 384 + 4 * source rank)
            if fx.thread_idx.x == 0:
                xw = arg_flag + fx.Int64((fx.block_idx.x % 8) * 32 + 64)
                t1 = flag_add(xw, 1)
                if t1 == N_BLOCKS // 8 - 1:
                    flag_add(xw, -(N_BLOCKS // 8))
                    t2 = flag_add(arg_flag + 4, 1)
                    if t2 == 7:
                        flag_add(arg_flag + 4, -8)
                        seq = fx.Int32(counter[0]) + 1
                        counter[0] = seq
                        for d in range_constexpr(8):
                            fr = buffer_ops.create_buffer_resource_from_addr(fx.Int64(peers[FLAGS + d]),
                                                                           num_records_bytes=512)
                            buffer_ops.buffer_store(seq, fr, 96 + i32_rank, cache_modifier=SYS_CM)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, arg_indices: fx.Tensor, i32_xrows: fx.Int32,
               i32_m: fx.Int32, counter: fx.Tensor, arg_flag: fx.Int64, peers: fx.Tensor, counts: fx.Tensor,
               i32_rank: fx.Int32, i32_buf: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, arg_indices, i32_xrows, i32_m, counter, arg_flag, peers, counts, i32_rank,
               i32_buf).launch(grid=(N_BLOCKS, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch
