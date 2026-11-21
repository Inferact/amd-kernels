# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from AMD-AGI/AKLR BF16 decode kernels (MIT), through aiter `ops/flydsl/kernels/gemm_a16w16_band_gfx950.py`.
"""Split-K band-stream bf16 GEMM, N 20480 x K 7168, 97-256 rows: the lm_head shard at the row counts where the
activation stream through the L2 binds the band kernel (`band.py`).

A group of S = 2 workgroups on one XCD owns 160 consecutive vocabulary columns; member i streams the K slice
[i K / 2, (i + 1) K / 2) of the weights of all 160 columns and of every activation row, so a workgroup reads
(M + 160) K / 2 operands instead of (M + 80) K: its activation traffic halves. The partial tiles are summed by a
fixed-order exchange through a workspace (no atomics on the output): every wave publishes the fp32 fragments of the row
blocks owned by the other member and raises a flag, then adds what the other member published for its own row blocks
(row block u of a wave is owned by member (u + i) mod S in member i's local order, so every register index is static);
the owner stages the rows in LDS and stores (or pushes) them. The sum runs in member order 0 .. S - 1 for every row,
independent of the owner, the batch size and the row's position. The flags re-arm themselves (the reader clears
what it consumed), so the workspace needs no reset between launches and serves one stream at a time.
The activation and weight tiles have their own LDS rings (A: dA + 1 stages, B: dB + 1; the weight ring is the deeper one
because its loads come from HBM): iteration t waits for A(t) and B(t) only, so the loads issued after them stay in
flight (`_vm_waits`).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T, Vector

from .band import _OOB, _OOB_ELEMS, FLAGS, RECV_ROWS, V, _as_i32, _bstore, _gep, _lds_ptr3, _swizzle_xor16
from .flags import SYS_CM, flag_add
from .reference import K, N

# (max rows, S, M waves, N waves, BLOCK_K, A distance, B distance, K stagger)
SPLITK = (
    (128, 2, 2, 2, 64, 2, 3, 1),
    (192, 2, 2, 2, 64, 2, 3, 1),
    (256, 2, 2, 2, 64, 1, 3, 1),
)
WS_POLICY = 0x10  # device scope (sc1): the exchange stays right if a group's members land on different XCDs
FLAG_STRIDE = 32  # int32 words between flags (one 128 B line each)


def workspace_words(cfg):
    """float32 words of the exchange workspace and int32 words of its flags for one device."""
    rows, S, MW, NW = cfg[:4]
    frags = (rows // 16 // MW // S) * (80 * S // 16 // NW)
    slots = (256 // S) * S * S * MW * NW
    return slots * frags * 64 * 4, slots * FLAG_STRIDE


def _vm_waits(loops, d_a, d_b, la, lb):
    """vmcnt each iteration may leave outstanding: the loads issued after the later of A(t), B(t) in issue order
    (iteration t issues A(t + d_a), then B(t + d_b); the prologue issues what iterations -d_b .. -1 would)."""
    seq = []
    for t in range(-d_b, loops):
        if 0 <= t + d_a < loops:
            seq.append((t, ("a", t + d_a)))
        if 0 <= t + d_b < loops:
            seq.append((t, ("b", t + d_b)))
    pos, cnt = {}, 0
    for _, op in seq:
        cnt += la if op[0] == "a" else lb
        pos[op] = cnt
    waits = []
    for t in range(loops):
        before = sum(la if op[0] == "a" else lb for u, op in seq if u < t)
        waits.append(before - max(pos[("a", t)], pos[("b", t)]))
    return waits


def compile_splitk_gemm(cfg: tuple, push: str | None = None, gather: bool = False):
    """The launcher of one row band; ``gather``: the activation rows come through a device row map; ``push``: the
    row-mapped GEMM sending its rows to ``owner`` ranks or ``all`` (implies the row map)."""
    rows, S, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K, D_A, D_B, K_STAGGER = cfg
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
    assert k % (S * BLOCK_K) == 0 and 1 <= D_A <= D_B
    NA, NB = D_A + 1, D_B + 1
    K_T = k // S
    BLOCK_K_LOOPS = K_T // BLOCK_K
    assert BLOCK_K_LOOPS > D_B
    WARP_K_STEPS = BLOCK_K // WMMA_K
    NUM_WAVES = BLOCK_M_WARPS * BLOCK_N_WARPS
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    TILE_M = rows
    TILE_N = 80 * S
    GPX = 256 // S // 8  # groups per XCD
    WARP_M_STEPS = TILE_M // BLOCK_M_WARPS // WMMA_M
    WARP_N_STEPS = TILE_N // BLOCK_N_WARPS // WMMA_N
    assert WARP_M_STEPS * WMMA_M * BLOCK_M_WARPS == TILE_M and WARP_N_STEPS * WMMA_N * BLOCK_N_WARPS == TILE_N
    assert WARP_M_STEPS % S == 0
    WARP_M = WARP_M_STEPS * WMMA_M
    WARP_N = WARP_N_STEPS * WMMA_N
    OWN_STEPS = WARP_M_STEPS // S  # row blocks per wave this member finalizes
    SLOT = OWN_STEPS * WARP_N_STEPS * WARP_SIZE * WMMA_C_FRAG_VALUES  # float32 words per (source, destination, wave)
    WS_WORDS, FLAG_WORDS = workspace_words(cfg)
    BLOCK_K_BYTES = BLOCK_K * DTYPE_BYTES
    A_STAGE_ELEMS = TILE_M * BLOCK_K
    B_STAGE_ELEMS = TILE_N * BLOCK_K
    AS_ELEMS = NA * A_STAGE_ELEMS
    LDS_ELEMS = AS_ELEMS + NB * B_STAGE_ELEMS
    assert LDS_ELEMS * DTYPE_BYTES <= 160 * 1024
    OWN_ROWS = TILE_M // S
    C_STRIDE = TILE_N + LDG_VEC_SIZE  # the staged output rows, padded by one store vector
    assert OWN_ROWS * C_STRIDE <= AS_ELEMS
    LDG_ASYNC_VEC_SIZE = DMA_BYTES // DTYPE_BYTES
    LDG_X_THREADS_AS = BLOCK_K // LDG_ASYNC_VEC_SIZE
    ROWS_PER_LOAD = BLOCK_THREADS // LDG_X_THREADS_AS
    assert TILE_M % ROWS_PER_LOAD == 0 and TILE_N % ROWS_PER_LOAD == 0
    A_LOADS = TILE_M // ROWS_PER_LOAD
    B_LOADS = TILE_N // ROWS_PER_LOAD
    VM_WAITS = _vm_waits(BLOCK_K_LOOPS, D_A, D_B, A_LOADS, B_LOADS)
    assert len(set(VM_WAITS[:BLOCK_K_LOOPS - D_B])) == 1 and max(VM_WAITS) < VMCNT_MAX
    LdsArr = fx.Array[fx.BFloat16, LDS_ELEMS, 16]
    assert push in (None, "owner", "all")
    rowmap = gather or bool(push)
    name = (f"bf16_gemm_n20480_k7168_splitk_tm{TILE_M}_bk{BLOCK_K}_d{D_A}{D_B}" + (f"_push_{push}" if push else "")
            + ("_gather" if gather and not push else ""))

    @flyc.kernel(name=name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_y: fx.Int64, arg_indices: fx.Tensor, i32_xrows: fx.Int32,
               i32_m: fx.Int32, counter: fx.Tensor, arg_flag: fx.Int64, peers: fx.Tensor, counts: fx.Tensor,
               i32_rank: fx.Int32, i32_buf: fx.Int32, arg_ws: fx.Int64, arg_sflags: fx.Int64):
        acc_init = arith.constant_vector(0.0, T.vec(WMMA_C_FRAG_VALUES, T.f32))
        bf16x8_ty = fx.Vector.make_type(LDG_VEC_SIZE, fx.BFloat16)
        a_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_xrows * (k * DTYPE_BYTES))
        b_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=n * k * DTYPE_BYTES)
        c_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_y, num_records_bytes=i32_m * (n * DTYPE_BYTES))
        ws_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_ws, num_records_bytes=WS_WORDS * 4)
        sf_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_sflags, num_records_bytes=FLAG_WORDS * 4)
        lds_base = fx.SharedAllocator().allocate(LdsArr).peek().ptr
        b_lds_ptr = lds_base + AS_ELEMS
        a_lds_i64 = fx.Int64(fx.ptrtoint(lds_base))
        b_lds_i64 = a_lds_i64 + fx.Int64(AS_ELEMS * DTYPE_BYTES)

        def as_load(stage, row, col):
            elem_off = fx.Int64(stage) * A_STAGE_ELEMS + fx.Int64(row) * BLOCK_K + fx.Int64(col)
            return fx.ptr_load(lds_base + elem_off, result_type=bf16x8_ty)

        def bs_load(stage, row, col):
            elem_off = fx.Int64(stage) * B_STAGE_ELEMS + fx.Int64(row) * BLOCK_K + fx.Int64(col)
            return fx.ptr_load(b_lds_ptr + elem_off, result_type=bf16x8_ty)

        tid = fx.thread_idx.x
        wid = tid // WARP_SIZE
        w_tid = tid % WARP_SIZE
        pid = fx.block_idx.x
        # a group's members are consecutive workgroups of one XCD (workgroup i runs on XCD i mod 8): a speed choice only
        me = fx.Int32((pid // 8) % S)
        gid = fx.Int32((pid % 8) * GPX + (pid // 8) // S)
        n_base = gid * TILE_N
        k_base = me * K_T
        k_blocks16 = fx.Int32(BLOCK_K_BYTES // 16)
        wm = wid // BLOCK_N_WARPS
        wn = wid % BLOCK_N_WARPS
        warp_m_idx = wm * WARP_M
        warp_n_idx = wn * WARP_N
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

        # tile row (LDS) -> activation row of this member: row block (u // S) S + (u + me) mod S above the wave's first
        # block, u = the block's index in the wave (the blocks of the S members tile the wave's blocks once); rows past
        # the count read zeros (an offset past the bound)
        a_row_off, a_col_sw = [], []
        for ii in range_constexpr(A_LOADS):
            global_tid = BLOCK_THREADS * ii + tid
            lr = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            a_col_sw.append(_swizzle_xor16(lr, k_local_idx * DTYPE_BYTES, k_blocks16))
            tl = lr // WMMA_M
            u = tl % WARP_M_STEPS
            gb = tl - u + (u // S) * S + (u % S + me) % S
            grow = gb * WMMA_M + lr % WMMA_M
            live = grow < i32_m
            if const_expr(rowmap):
                src_row = fx.Int32(arg_indices[live.select(grow, fx.Int32(0))])
            else:
                src_row = grow
            a_row_off.append(live.select(src_row * (k * DTYPE_BYTES) + k_base * DTYPE_BYTES, fx.Int32(_OOB)))
        b_row_off, b_col_sw = [], []
        for ii in range_constexpr(B_LOADS):
            global_tid = BLOCK_THREADS * ii + tid
            lr = global_tid // LDG_X_THREADS_AS
            k_local_idx = global_tid % LDG_X_THREADS_AS * LDG_ASYNC_VEC_SIZE
            b_col_sw.append(_swizzle_xor16(lr, k_local_idx * DTYPE_BYTES, k_blocks16))
            b_row_off.append((n_base + lr) * (k * DTYPE_BYTES) + k_base * DTYPE_BYTES)

        def ldg_sts_a_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_offset = _as_i32(a_row_off[ii] + fx.Int32(k_offset) * DTYPE_BYTES + a_col_sw[ii])
            if const_expr(lds_ptr is None):
                stage_off = arith.index_cast(T.i64, fx.Index(write_stage) * (A_STAGE_ELEMS * DTYPE_BYTES))
                lds_ptr = _gep(_lds_ptr3(a_lds_i64, fx.Int64(stage_off)), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            buffer_load_lds_inline(a_rsrc, lds_ptr, global_offset, bits="sc0")
            return lds_ptr

        def ldg_sts_b_async_one(ii, k_offset, write_stage, lds_ptr=None):
            global_offset = _as_i32(b_row_off[ii] + fx.Int32(k_offset) * DTYPE_BYTES + b_col_sw[ii])
            if const_expr(lds_ptr is None):
                stage_off = arith.index_cast(T.i64, fx.Index(write_stage) * (B_STAGE_ELEMS * DTYPE_BYTES))
                lds_ptr = _gep(_lds_ptr3(b_lds_i64, fx.Int64(stage_off)), warp_offset)
            else:
                lds_ptr = _gep(lds_ptr, BLOCK_THREADS * DMA_BYTES)
            # B is read once; a non-temporal load keeps it from displacing the reused A lines
            buffer_load_lds_inline(b_rsrc, lds_ptr, global_offset, bits="sc0 nt")
            return lds_ptr

        def issue_a(k_offset, stage):
            lds_ptr = None
            for i in range_constexpr(A_LOADS):
                lds_ptr = ldg_sts_a_async_one(i, k_offset, stage, lds_ptr)

        def issue_b(k_offset, stage):
            lds_ptr = None
            for i in range_constexpr(B_LOADS):
                lds_ptr = ldg_sts_b_async_one(i, k_offset, stage, lds_ptr)

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

        def frag_loads(sa, sb, kk):
            wk = kk * WMMA_K
            a_f = [load_a_frag(sa, ii, wk) for ii in range_constexpr(WARP_M_STEPS)]
            b_f = [load_b_frag(sb, ii, wk) for ii in range_constexpr(WARP_N_STEPS)]
            return a_f, b_f

        def pipe_compute(a_stage, b_stage, c_frags, dma_a=None, dma_b=None):
            """One K block; dma_a / dma_b = (k offset, stage) of the A / B loads issued between the K steps."""
            sa, sb = fx.Index(a_stage), fx.Index(b_stage)
            c_new = list(c_frags)
            items = ([("a", i) for i in range(A_LOADS)] if dma_a is not None else []) + (
                [("b", i) for i in range(B_LOADS)] if dma_b is not None else [])
            nd = len(items)
            bounds = [min(nd, (nd * j + WARP_K_STEPS - 1) // WARP_K_STEPS) for j in range(WARP_K_STEPS + 1)]
            ptrs = {"a": None, "b": None}
            cur = frag_loads(sa, sb, 0)
            for kk in range_constexpr(WARP_K_STEPS):
                for j in range_constexpr(bounds[kk], bounds[kk + 1]):
                    kind, i = items[j]
                    if const_expr(kind == "a"):
                        ptrs["a"] = ldg_sts_a_async_one(i, dma_a[0], dma_a[1], ptrs["a"])
                    else:
                        ptrs["b"] = ldg_sts_b_async_one(i, dma_b[0], dma_b[1], ptrs["b"])
                nxt = None
                if const_expr(kk + 1 < WARP_K_STEPS):
                    nxt = frag_loads(sa, sb, kk + 1)
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
            return (off >= fx.Int32(K_T)).select(off - fx.Int32(K_T), off)

        # staggered K starts spread the concurrent reads of the activation over L2 channels
        ks_begin = fx.Int32(pid * K_STAGGER % BLOCK_K_LOOPS * BLOCK_K)
        for t in range_constexpr(-D_B, 0):
            if const_expr(t + D_A >= 0):
                issue_a(kwrap(ks_begin + (t + D_A) * BLOCK_K), (t + D_A) % NA)
            if const_expr(t + D_B >= 0):
                issue_b(kwrap(ks_begin + (t + D_B) * BLOCK_K), (t + D_B) % NB)
        rocdl.sched_barrier(0)

        init_state = [ks_begin, arith.constant(0, index=True), arith.constant(0, index=True)] + c_frags
        for bki, state in range(0, BLOCK_K_LOOPS - D_B, 1, init=init_state):
            k_offset = state[0]
            a_stage = fx.Index(state[1])
            b_stage = fx.Index(state[2])
            wait_barrier(VM_WAITS[0])
            c_frags_new = pipe_compute(a_stage, b_stage, state[3:],
                                       (kwrap(k_offset + D_A * BLOCK_K), (a_stage + D_A) % NA),
                                       (kwrap(k_offset + D_B * BLOCK_K), (b_stage + D_B) % NB))
            results = yield [kwrap(k_offset + fx.Int32(BLOCK_K)), (a_stage + 1) % NA, (b_stage + 1) % NB] + c_frags_new
        k_offset = results[0]
        a_stage = fx.Index(results[1])
        b_stage = fx.Index(results[2])
        c_frags = results[3:]
        for j in range_constexpr(D_B):
            t = BLOCK_K_LOOPS - D_B + j
            wait_barrier(VM_WAITS[t])
            dma_a = (kwrap(k_offset + D_A * BLOCK_K), (a_stage + D_A) % NA) if t + D_A < BLOCK_K_LOOPS else None
            c_frags = pipe_compute(a_stage, b_stage, c_frags, dma_a)
            k_offset = kwrap(k_offset + fx.Int32(BLOCK_K))
            a_stage = (a_stage + 1) % NA
            b_stage = (b_stage + 1) % NB

        def flag_idx(dest, src):
            return ((gid * S + dest) * S + src) * NUM_WAVES * FLAG_STRIDE + wid * FLAG_STRIDE

        def ws_base(dest, src):
            return (((gid * S + dest) * S + src) * NUM_WAVES + wid) * SLOT

        for q in range_constexpr(1, S):
            base = ws_base((me + q) % S, me)
            for u in range_constexpr(WARP_M_STEPS):
                if const_expr(u % S == q):
                    for v in range_constexpr(WARP_N_STEPS):
                        f = (u // S) * WARP_N_STEPS + v
                        buffer_ops.buffer_store(c_frags[u * WARP_N_STEPS + v], ws_rsrc,
                                                base + (f * WARP_SIZE + w_tid) * WMMA_C_FRAG_VALUES,
                                                cache_modifier=WS_POLICY)
        fx.rocdl.s_waitcnt(vmcnt=0)
        buffer_ops.buffer_store(fx.Int32(1), sf_rsrc,
                                (w_tid < S - 1).select(flag_idx((me + w_tid + 1) % S, me), fx.Int32(_OOB_ELEMS)),
                                cache_modifier=SYS_CM)
        # lane l (< S - 1) polls the flag of source (me - l - 1) mod S with a system-scope load, then re-arms it
        if w_tid < S - 1:
            fidx = flag_idx(me, (me + S - 1 - w_tid) % S)
            foff = fx.arith.unwrap(fx.Int32(fidx) * 4)

            def poll():
                return fx.Int32(llvm.InlineAsmOp(
                    T.i32, [foff, sf_rsrc], "buffer_load_dword $0, $1, $2, 0 offen sc0 sc1\n\ts_waitcnt vmcnt(0)",
                    "=v,v,s", has_side_effects=True).result)

            val = poll()
            while val < 1:
                val = poll()
            buffer_ops.buffer_store(fx.Int32(0), sf_rsrc, fidx, cache_modifier=SYS_CM)

        c_m_vec_idx = w_tid // WMMA_N * WMMA_C_FRAG_VALUES
        c_n_idx = w_tid % WMMA_N
        own = []
        for uo in range_constexpr(OWN_STEPS):
            for v in range_constexpr(WARP_N_STEPS):
                f = uo * WARP_N_STEPS + v
                mine = c_frags[uo * S * WARP_N_STEPS + v]
                terms = []
                for i in range_constexpr(S):
                    off = ws_base(me, fx.Int32(i)) + (f * WARP_SIZE + w_tid) * WMMA_C_FRAG_VALUES
                    got = buffer_ops.buffer_load(ws_rsrc, off, vec_width=WMMA_C_FRAG_VALUES, dtype=fx.Float32,
                                                 mask=(me != i), cache_modifier=WS_POLICY)
                    terms.append(std_arith.SelectOp(fx.arith.unwrap(me == i), fx.arith.unwrap(mine),
                                                    fx.arith.unwrap(got)).result)
                acc = terms[0]
                for i in range_constexpr(1, S):
                    acc = std_arith.AddFOp(acc, terms[i]).result
                own.append(acc)
        # the A ring is reused for the staged output
        fx.gpu.barrier()
        for uo in range_constexpr(OWN_STEPS):
            for v in range_constexpr(WARP_N_STEPS):
                cast = Vector(own[uo * WARP_N_STEPS + v]).to(fx.BFloat16)
                lds_m = fx.Int64((wm * OWN_STEPS + uo) * WMMA_M + c_n_idx)
                lds_n = fx.Int64(warp_n_idx + v * WMMA_N + c_m_vec_idx)
                fx.ptr_store(cast, lds_base + (lds_m * C_STRIDE + lds_n))
        fx.gpu.barrier()

        if const_expr(push):
            dst = [buffer_ops.create_buffer_resource_from_addr(fx.Int64(peers[i32_buf * 8 + d]),
                                                           num_records_bytes=RECV_ROWS * V * 2)
                   for d in range_constexpr(8)]
            cum = [fx.Int32(0)]
            for d in range_constexpr(8):
                cum.append(cum[d] + fx.Int32(counts[d]))

        XT = TILE_N // LDG_VEC_SIZE
        NV = OWN_ROWS * XT
        for i in range_constexpr((NV + BLOCK_THREADS - 1) // BLOCK_THREADS):
            global_tid = BLOCK_THREADS * i + tid
            lm = global_tid // XT
            n_local = global_tid % XT * LDG_VEC_SIZE

            def store_c(lm=lm, n_local=n_local):
                vec = fx.ptr_load(lds_base + (fx.Int64(lm) * C_STRIDE + fx.Int64(n_local)), result_type=bf16x8_ty)
                cb = lm // WMMA_M
                gb = (cb // OWN_STEPS) * WARP_M_STEPS + (cb % OWN_STEPS) * S + me
                r = gb * WMMA_M + lm % WMMA_M
                col = n_base + n_local
                live = r < i32_m
                if const_expr(push):
                    # one 16-byte vector of output row r at column col of this rank's 20480, to the row's owner (counts:
                    # this rank's rows are sorted by owner) or to every rank; rows past the count and every other
                    # destination take an offset past the bound, a dropped store
                    if const_expr(push == "owner"):
                        owner = fx.Int32(0)
                        lo = fx.Int32(0)
                        for j in range_constexpr(1, 8):
                            ge = r >= cum[j]
                            owner = ge.select(fx.Int32(j), owner)
                            lo = ge.select(cum[j], lo)
                        off = (r - lo) * V + i32_rank * n + col
                    else:
                        off = r * V + i32_rank * n + col
                    for d in range_constexpr(8):
                        ok = (live & (owner == d)) if const_expr(push == "owner") else live
                        _bstore(vec, dst[d], ok.select(off, fx.Int32(_OOB_ELEMS)), SYS_CM)
                else:
                    _bstore(vec, c_rsrc, live.select(r * n + col, fx.Int32(_OOB_ELEMS)), 0)

            if const_expr(NV % BLOCK_THREADS != 0 and i == NV // BLOCK_THREADS):
                if fx.Int32(global_tid) < fx.Int32(NV):
                    store_c()
            else:
                store_c()

        if const_expr(push):
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            # the band kernel's two-level ticket: the last arriver stores the sequence into this rank's word of every
            # receiver's flag block (byte 384 + 4 * source rank)
            if fx.thread_idx.x == 0:
                xw = arg_flag + fx.Int64((fx.block_idx.x % 8) * 32 + 64)
                t1 = flag_add(xw, 1)
                if t1 == 256 // 8 - 1:
                    flag_add(xw, -(256 // 8))
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
               i32_rank: fx.Int32, i32_buf: fx.Int32, arg_ws: fx.Int64, arg_sflags: fx.Int64, stream: fx.Stream):
        kernel(arg_x, arg_w, arg_y, arg_indices, i32_xrows, i32_m, counter, arg_flag, peers, counts, i32_rank,
               i32_buf, arg_ws, arg_sflags).launch(grid=(256, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch
