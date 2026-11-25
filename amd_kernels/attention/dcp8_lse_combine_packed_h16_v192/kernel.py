# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# The hand-off protocol (uncached IPC pools and control blocks, per-workgroup flag and start words, the counter bumped
# by the last workgroup) follows attention/dcp8_lse_combine_vup_mla_h12_kv512/kernel.py.
"""DCP8 combine of packed partials in one launch: push each owner's rows, LSE-weighted combine at the owner.

A call takes ``seq = counter + 1`` and the slot ``seq & 1`` of every pool (``[8 sources, max_tokens, 48, 66]`` bf16).

Push workgroups (8 per 16-row tile, the destination rotated by the rank): copy the tile's rows of ``send[d]`` (6336 B
each, contiguous) into source ``rank``'s rows of rank d's slot with system-scope stores, wait for them, then store
``seq`` into word ``rank * MAX_TILES + tile`` of rank d's flag array: no atomics. Rank ``rank`` is its own destination.

Combine workgroups (after the push workgroups in the grid, so they never starve them; 6 per tile, 8 sub-heads each):
wait for the 8 flags of their tile, read the 8 sources' rows (local uncached memory) and combine them (base-2 LSEs) as vLLM's
``_dcp_a2a_unpack_combine`` does: the max LSE, ``g = log2(sum 2^(l - max)) + max``, weights ``2^(l - g)``, products
summed in source order in fp32 (multiplies and adds, not fused), rounded once. Every workgroup stores ``seq`` into its start
word after reading the counter; one extra workgroup (dispatched last) waits for every start word and then bumps the
counter, so a CUDA graph replays with no host state and the row count may change between calls. A slot is rewritten
two calls later, after every peer's flags of the call in between, which it sets after its own previous call finished.

Control block (int32 words): flags [0, 8 * MAX_TILES), counter SEQ_WORD, start words START_WORD + workgroup.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import D, PACK, SUB, WORLD

NT = 256
TILE = 16
MAX_TILES = 32
MAX_TOKENS = TILE * MAX_TILES
GROUPS = 6
SUB_PER_GROUP = SUB // GROUPS
ROW_DW = SUB * PACK // 2
PUSH_PASSES = -(-TILE * ROW_DW // (NT * 4))
COMBINE_PASSES = TILE * SUB_PER_GROUP * (D // 8) // NT
SYS = 0x11
OOB = 0x3FFFFFF0
CTL_BYTES = 8192
FLAG_WORD, SEQ_WORD, START_WORD = 0, 4 * MAX_TILES * WORLD // 4 + 64, 512
assert START_WORD + (WORLD + GROUPS) * MAX_TILES + 1 <= CTL_BYTES // 4


def _ptr(elem, addr_i64, align):
    ty = fx.PointerType.get(elem, address_space=fx.AddressSpace.Global, alignment=align)
    return fx.inttoptr(ty, fx.Int64(addr_i64))


def _volatile_load(addr_i64):
    """A system-coherent load the compiler neither hoists out of a spin loop nor drops."""
    return fx.Int32(_llvm.LoadOp(T.i32, _ptr(T.i32, addr_i64, 4).llvm_ptr, alignment=4, volatile_=True).result)


def _volatile_load_v4(addr_i64):
    ty = ir.VectorType.get([4], T.i32)
    return Vec(_llvm.LoadOp(ty, _ptr(T.i32, addr_i64, 16).llvm_ptr, alignment=16, volatile_=True).result)


def _pk_bf16(a, b):
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def grid_of(tiles):
    return (WORLD + GROUPS) * tiles + 1


@launcher
def compile_combine():
    """The single variant: the row count is a runtime argument."""
    name = "dcp8_lse_combine_packed_h16_v192"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_send: fx.Int64,  # [8, T, 48, 66] bf16, rows contiguous
        i64_send_stride: fx.Int64,  # bytes between destinations
        arg_out: fx.Int64,  # [T, 48, 64] bf16
        arg_tab: fx.Int64,  # int64 [16]: pool and control block of each rank
        i32_m: fx.Int32,
        i32_rank: fx.Int32,
        i32_mmax: fx.Int32,
        i64_slot_bytes: fx.Int64,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        tab = _ptr(T.i64, arg_tab, 8)
        tiles = (i32_m + (TILE - 1)) // TILE
        n_push = tiles * WORLD
        last = tiles * (WORLD + GROUPS)
        me_ctl = fx.Int64(tab[WORLD + i32_rank])
        ctl = buffer_ops.create_buffer_resource_from_addr(me_ctl, num_records_bytes=CTL_BYTES)
        slot_records = i32_mmax * (WORLD * ROW_DW * 4)

        def read_seq():
            return fx.Int32(
                fx.rocdl.readfirstlane(
                    T.i32,
                    fx.Int32(buffer_ops.buffer_load(ctl, SEQ_WORD, vec_width=1, dtype=fx.Int32, cache_modifier=SYS)) + 1,
                )
            )

        if bid < n_push:
            t = bid // WORLD
            d = (i32_rank + bid % WORLD) % WORLD
            seq = read_seq()
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, START_WORD + bid, cache_modifier=SYS)
            src = buffer_ops.create_buffer_resource_from_addr(
                arg_send + fx.Int64(d) * i64_send_stride, num_records_bytes=i32_m * (ROW_DW * 4)
            )
            dst_base = fx.Int64(tab[d]) + fx.Int64(seq & 1) * i64_slot_bytes
            dst = buffer_ops.create_buffer_resource_from_addr(dst_base, num_records_bytes=slot_records)
            n_dw = ((i32_m - t * TILE) < TILE).select(i32_m - t * TILE, fx.Int32(TILE)) * ROW_DW
            row0 = t * (TILE * ROW_DW)
            dst0 = (i32_rank * i32_mmax) * ROW_DW + row0
            vals = []
            for p in range_constexpr(PUSH_PASSES):
                e = (p * NT + tx) * 4
                vals.append(Vec(buffer_ops.buffer_load(src, row0 + e, vec_width=4, dtype=fx.Int32, mask=e < n_dw)))
            for p in range_constexpr(PUSH_PASSES):
                e = (p * NT + tx) * 4
                buffer_ops.buffer_store(vals[p], dst, (e < n_dw).select(dst0 + e, fx.Int32(OOB)), cache_modifier=SYS)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                fr = buffer_ops.create_buffer_resource_from_addr(fx.Int64(tab[WORLD + d]), num_records_bytes=CTL_BYTES)
                buffer_ops.buffer_store(seq, fr, FLAG_WORD + i32_rank * MAX_TILES + t, cache_modifier=SYS)

        if bid == last:
            seq = read_seq()
            for k in range_constexpr(-(-(WORLD + GROUPS) * MAX_TILES // (NT * 4))):
                wid = (tx + k * NT) * 4
                lane_ok = wid < last
                addr = me_ctl + (START_WORD + lane_ok.select(wid, fx.Int32(0))) * 4
                fv = _volatile_load_v4(addr)
                pending = lane_ok & (
                    ((fv[0] - seq) < 0)
                    | ((wid + 1 < last) & ((fv[1] - seq) < 0))
                    | ((wid + 2 < last) & ((fv[2] - seq) < 0))
                    | ((wid + 3 < last) & ((fv[3] - seq) < 0))
                )
                while pending:
                    fv = _volatile_load_v4(addr)
                    pending = lane_ok & (
                        ((fv[0] - seq) < 0)
                        | ((wid + 1 < last) & ((fv[1] - seq) < 0))
                        | ((wid + 2 < last) & ((fv[2] - seq) < 0))
                        | ((wid + 3 < last) & ((fv[3] - seq) < 0))
                    )
            fx.gpu.barrier()
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, SEQ_WORD, cache_modifier=SYS)

        if (bid >= n_push) & (bid < last):
            c = bid - n_push
            t = c // GROUPS
            grp = c % GROUPS
            seq = read_seq()
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, START_WORD + bid, cache_modifier=SYS)
            if tx < WORLD:
                addr = me_ctl + fx.Int64((FLAG_WORD + tx * MAX_TILES + t) * 4)
                fv = _volatile_load(addr)
                while (fv - seq) < 0:
                    fv = _volatile_load(addr)
            fx.gpu.barrier()
            mine = fx.Int64(tab[i32_rank]) + fx.Int64(seq & 1) * i64_slot_bytes
            pool = buffer_ops.create_buffer_resource_from_addr(mine, num_records_bytes=slot_records)
            outr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_m * (SUB * D * 2))
            k8 = tx % 8
            neg_inf = fx.Float32(float("-inf"))
            for p in range_constexpr(COMBINE_PASSES):
                u = p * (NT // 8) + tx // 8
                row = t * TILE + u // SUB_PER_GROUP
                sub = grp * SUB_PER_GROUP + u % SUB_PER_GROUP
                ok = row < i32_m
                xs, ls = [], []
                for s in range_constexpr(WORLD):
                    base = (s * i32_mmax + row) * ROW_DW + sub * (PACK // 2)
                    xs.append(
                        Vec(buffer_ops.buffer_load(pool, base + k8 * 4, vec_width=4, dtype=fx.Int32, mask=ok, cache_modifier=SYS))
                    )
                    lv = fx.Int32(
                        buffer_ops.buffer_load(pool, base + D // 2, vec_width=1, dtype=fx.Int32, mask=ok, cache_modifier=SYS)
                    ).bitcast(fx.Float32)
                    ls.append(((lv != lv) | (lv == fx.Float32(float("inf")))).select(neg_inf, lv))
                m = ls[0]
                for s in range_constexpr(1, WORLD):
                    m = (ls[s] > m).select(ls[s], m)
                m = (m == neg_inf).select(fx.Float32(0.0), m)
                tot = fx.Float32(0.0)
                for s in range_constexpr(WORLD):
                    tot = tot + fx.Float32(fx.math.exp2(ls[s] - m))
                g = fx.Float32(fx.math.log2(tot)) + m
                acc = [fx.Float32(0.0)] * 8
                for s in range_constexpr(WORLD):
                    ws = fx.Float32(fx.math.exp2(ls[s] - g))
                    ws = (ws != ws).select(fx.Float32(0.0), ws)
                    zero = ws == fx.Float32(0.0)
                    for q in range_constexpr(4):
                        w32 = fx.Int32(xs[s][q])
                        lo = (w32 << 16).bitcast(fx.Float32)
                        hi = (w32 & fx.Int32(-65536)).bitcast(fx.Float32)
                        acc[2 * q] = acc[2 * q] + zero.select(fx.Float32(0.0), lo) * ws
                        acc[2 * q + 1] = acc[2 * q + 1] + zero.select(fx.Float32(0.0), hi) * ws
                pk = [_pk_bf16(acc[2 * q], acc[2 * q + 1]) for q in range_constexpr(4)]
                buffer_ops.buffer_store(
                    Vec.from_elements(pk, fx.Int32), outr, (row * SUB + sub) * (D // 2) + k8 * 4, mask=ok
                )

    @flyc.jit
    def launch(
        arg_send: fx.Int64,
        i64_send_stride: fx.Int64,
        arg_out: fx.Int64,
        arg_tab: fx.Int64,
        i32_m: fx.Int32,
        i32_rank: fx.Int32,
        i32_mmax: fx.Int32,
        i64_slot_bytes: fx.Int64,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_send, i64_send_stride, arg_out, arg_tab, i32_m, i32_rank, i32_mmax, i64_slot_bytes).launch(
            grid=(i32_grid, 1, 1), block=(NT, 1, 1), stream=stream
        )

    launch.kernel_name = name
    return launch
