# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""FlyDSL direct peer push for a DCP8 ShardQ context workspace.

Rank ``r`` owns global token ``t`` when ``t % 8 == r`` and stores it at
local row ``t // 8``.  One workgroup moves eight local rows, loading each MLA
and indexer byte once and writing it directly to the dense workspace of all
eight ranks.  There is no packed staging buffer, RCCL all-gather, or scatter.

The production variant uses device-resident epochs over uncached IPC memory:
all ranks first announce that the previous workspace has been consumed, then
push, and the last workgroup announces completion.  This is graph safe and
does not require a host collective per layer.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.common.mono.ops import CM_SYS
from amd_kernels.warm_up import launcher
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

WORLD = 8
PAGE = 64
ROWS_PER_CTA = 8
NTHREADS = 128
LINE_WORDS = 32
FREE_OFF = 0
READY_OFF = 1024
DATA_OFF = 4096
EXCHANGE_OFF = 2048


@launcher
def compile_exchange_barrier():
    """One CTA per GPU rendezvous on the current stream, also in graph replay."""

    @flyc.kernel(known_block_size=[64, 1, 1])
    def barrier(
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        rank: fx.Int32,
        state: fx.Int64,
    ):
        lane = fx.thread_idx.x
        bases = (b0, b1, b2, b3, b4, b5, b6, b7)
        epoch = _flag_load(state) + 1
        if lane < WORLD:
            _flag_store(
                _sel8(lane, bases) + EXCHANGE_OFF + fx.Int64(rank * LINE_WORDS * 4),
                epoch,
            )
            _spin(
                _sel8(rank, bases) + EXCHANGE_OFF + fx.Int64(lane * LINE_WORDS * 4),
                epoch,
            )
        fx.gpu.barrier()
        if lane == 0:
            _flag_store(state, epoch)

    @flyc.jit
    def launch(
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        rank: fx.Int32,
        state: fx.Int64,
        stream: fx.Stream,
    ):
        barrier(b0, b1, b2, b3, b4, b5, b6, b7, rank, state).launch(
            grid=(1, 1, 1), block=(64, 1, 1), stream=stream
        )

    launch.kernel_name = "dcp8_shardq_exchange_barrier"
    return launch


def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def _flag_store(addr, value):
    """System-scope write-through flag store to an IPC-mapped peer."""
    _llvm.StoreOp(
        fx.Int32(value).ir_value(),
        _gptr(addr).llvm_ptr,
        alignment=4,
        ordering=_llvm.AtomicOrdering.monotonic,
        syncscope="",
    )


def _flag_load(addr):
    return fx.Int32(
        _llvm.LoadOp(
            T.i32,
            _gptr(addr).llvm_ptr,
            alignment=4,
            ordering=_llvm.AtomicOrdering.monotonic,
            syncscope="",
        ).result
    )


@flyc.jit
def _spin(addr, expected):
    value = _flag_load(addr)
    while (value - expected) < 0:
        value = _flag_load(addr)


def _ticket_add(addr):
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(1).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def _sel8(index, values):
    out = values[7]
    for i in range(6, -1, -1):
        out = (index == i).select(values[i], out)
    return out


def _load_i32(resource, offset):
    return fx.Int32(
        buffer_ops.buffer_load(
            resource, offset, vec_width=1, dtype=fx.Int32, is_scalar=True
        )
    )


@launcher
def compile_context_push(
    mla_row_bytes: int,
    index_row_bytes: int,
    index_scale_bytes: int = 4,
    synchronize: bool = True,
    tiled_fp4: bool = False,
):
    """Compile one row-layout variant of the DCP8 direct push."""
    assert mla_row_bytes % 16 == 0
    assert index_row_bytes % 16 == 0
    assert index_scale_bytes == 4
    assert not tiled_fp4 or index_row_bytes == 64
    mla_page_bytes = PAGE * mla_row_bytes
    index_page_bytes = PAGE * (index_row_bytes + index_scale_bytes)
    scale_offset = PAGE * index_row_bytes
    mla_vecs = mla_row_bytes // 16
    index_vecs = index_row_bytes // 16
    name = (
        f"dcp8_context_push_mla{mla_row_bytes}_idx{index_row_bytes}"
        f"_{'sync' if synchronize else 'nosync'}_{'fp4' if tiled_fp4 else 'linear'}"
    )

    @flyc.kernel(name=name, known_block_size=[NTHREADS, 1, 1])
    def kernel(
        mla_base: fx.Int64,
        mla_block_stride: fx.Int64,
        mla_pages_per_block: fx.Int32,
        mla_block_table: fx.Int64,
        mla_bt_stride: fx.Int32,
        index_base: fx.Int64,
        index_block_stride: fx.Int64,
        index_block_table: fx.Int64,
        index_bt_stride: fx.Int32,
        seq_lens: fx.Int64,
        query_start_loc: fx.Int64,
        dst_block_table: fx.Int64,
        dst_bt_stride: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        workspace_bytes: fx.Int64,
        dst_index_offset: fx.Int64,
        rank: fx.Int32,
        max_local_pages: fx.Int32,
        num_reqs: fx.Int32,
        state: fx.Int64,
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        ctas_per_req = max_local_pages * (PAGE // ROWS_PER_CTA)
        req = bid // ctas_per_req
        within_req = bid - req * ctas_per_req
        page = within_req // (PAGE // ROWS_PER_CTA)
        row0 = (within_req % (PAGE // ROWS_PER_CTA)) * ROWS_PER_CTA
        total_ctas = num_reqs * ctas_per_req
        bases = (b0, b1, b2, b3, b4, b5, b6, b7)
        mine = _sel8(rank, bases)

        state_rsrc = buffer_ops.create_buffer_resource_from_addr(
            state, num_records_bytes=8
        )
        epoch = _load_i32(state_rsrc, 0)
        if synchronize:
            if (bid == 0) & (tx < WORLD):
                _flag_store(
                    _sel8(tx, bases) + FREE_OFF + fx.Int64(rank * (LINE_WORDS * 4)),
                    epoch + 1,
                )
            if tx < WORLD:
                _spin(
                    mine + FREE_OFF + fx.Int64(tx * (LINE_WORDS * 4)),
                    epoch + 1,
                )
            fx.gpu.barrier()

        seq_rsrc = buffer_ops.create_buffer_resource_from_addr(seq_lens)
        qsl_rsrc = buffer_ops.create_buffer_resource_from_addr(query_start_loc)
        mla_bt_rsrc = buffer_ops.create_buffer_resource_from_addr(mla_block_table)
        index_bt_rsrc = buffer_ops.create_buffer_resource_from_addr(index_block_table)
        dst_bt_rsrc = buffer_ops.create_buffer_resource_from_addr(dst_block_table)
        seq_len = _load_i32(seq_rsrc, req)
        query_len = _load_i32(qsl_rsrc, req + 1) - _load_i32(qsl_rsrc, req)
        context_len = seq_len - query_len
        index_rows = fx.max(seq_len - rank + (WORLD - 1), 0) // WORLD
        mla_rows = fx.max(context_len - rank + (WORLD - 1), 0) // WORLD
        first_row = page * PAGE + row0
        num_index = fx.max(fx.min(index_rows - first_row, ROWS_PER_CTA), 0)
        num_mla = fx.max(fx.min(mla_rows - first_row, ROWS_PER_CTA), 0)

        if num_index > 0:
            mla_block_col = page // mla_pages_per_block
            mla_block = _load_i32(mla_bt_rsrc, req * mla_bt_stride + mla_block_col)
            mla_src_addr = (
                mla_base
                + fx.Int64(mla_block) * mla_block_stride
                + fx.Int64(page % mla_pages_per_block) * mla_page_bytes
            )
            index_block = _load_i32(index_bt_rsrc, req * index_bt_stride + page)
            index_src_addr = index_base + fx.Int64(index_block) * index_block_stride
            dst_page0 = _load_i32(dst_bt_rsrc, req * dst_bt_stride)
            dst_page = dst_page0 + WORLD * page
            mla_src = buffer_ops.create_buffer_resource_from_addr(
                mla_src_addr, num_records_bytes=mla_page_bytes
            )
            index_src = buffer_ops.create_buffer_resource_from_addr(
                index_src_addr, num_records_bytes=index_page_bytes
            )
            mla_dsts = [
                buffer_ops.create_buffer_resource_from_addr(
                    base + DATA_OFF, num_records_bytes=workspace_bytes
                )
                for base in bases
            ]
            index_dsts = [
                buffer_ops.create_buffer_resource_from_addr(
                    base + DATA_OFF + dst_index_offset,
                    num_records_bytes=workspace_bytes - dst_index_offset,
                )
                for base in bases
            ]

            v = tx
            while v < num_mla * mla_vecs:
                local_row = row0 + v // mla_vecs
                col_vec = v % mla_vecs
                absolute_local_row = page * PAGE + local_row
                token = absolute_local_row * WORLD + rank
                value = Vec(
                    buffer_ops.buffer_load(
                        mla_src,
                        local_row * (mla_row_bytes // 4) + col_vec * 4,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                )
                dst_byte = (
                    (dst_page0 + token // PAGE) * mla_page_bytes
                    + (token % PAGE) * mla_row_bytes
                    + col_vec * 16
                )
                for d in range_constexpr(WORLD):
                    buffer_ops.buffer_store(
                        value,
                        mla_dsts[d],
                        dst_byte,
                        cache_modifier=CM_SYS,
                        offset_is_bytes=True,
                    )
                v += NTHREADS

            v = tx
            while v < num_index * index_vecs:
                local_row = row0 + v // index_vecs
                col_vec = v % index_vecs
                absolute_local_row = page * PAGE + local_row
                token = absolute_local_row * WORLD + rank
                src_vec = local_row * index_vecs + col_vec
                dst_vec = (token % PAGE) * index_vecs + col_vec
                if tiled_fp4:
                    src_vec = col_vec * PAGE + local_row
                    dst_vec = col_vec * PAGE + token % PAGE
                value = Vec(
                    buffer_ops.buffer_load(
                        index_src,
                        src_vec * 4,
                        vec_width=4,
                        dtype=fx.Int32,
                    )
                )
                dst_byte = (dst_page0 + token // PAGE) * index_page_bytes + dst_vec * 16
                for d in range_constexpr(WORLD):
                    buffer_ops.buffer_store(
                        value,
                        index_dsts[d],
                        dst_byte,
                        cache_modifier=CM_SYS,
                        offset_is_bytes=True,
                    )
                v += NTHREADS

            if tiled_fp4:
                # Each source owns two complete scale words per group in
                # this destination page. Assemble the four slot quarters
                # before the peer store, so ranks never race on a word.
                if tx < 8:
                    group = tx // 2
                    parity = tx % 2
                    word = fx.Int32(0)
                    for quarter in range_constexpr(4):
                        row = row0 + parity + 2 * quarter
                        src_word = _load_i32(
                            index_src, scale_offset // 4 + group * 16 + row % 16
                        )
                        byte = (src_word >> ((row // 16) * 8)) & 255
                        byte = (parity + 2 * quarter < num_index).select(byte, 0)
                        word = word | (byte << (quarter * 8))
                    token = first_row * WORLD + rank
                    dst_byte = (
                        (dst_page0 + token // PAGE) * index_page_bytes
                        + scale_offset
                        + group * PAGE
                        + (rank + parity * WORLD) * 4
                    )
                    for d in range_constexpr(WORLD):
                        buffer_ops.buffer_store(
                            word,
                            index_dsts[d],
                            dst_byte,
                            cache_modifier=CM_SYS,
                            offset_is_bytes=True,
                        )
            elif tx < num_index:
                local_row = row0 + tx
                absolute_local_row = page * PAGE + local_row
                token = absolute_local_row * WORLD + rank
                scale = buffer_ops.buffer_load(
                    index_src,
                    scale_offset // 4 + local_row,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                dst_byte = (
                    (dst_page0 + token // PAGE) * index_page_bytes
                    + scale_offset
                    + (token % PAGE) * index_scale_bytes
                )
                for d in range_constexpr(WORLD):
                    buffer_ops.buffer_store(
                        scale,
                        index_dsts[d],
                        dst_byte,
                        cache_modifier=CM_SYS,
                        offset_is_bytes=True,
                    )

        if synchronize:
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                old = _ticket_add(state + 4)
                if old + 1 == total_ctas:
                    for peer in range_constexpr(WORLD):
                        _flag_store(
                            bases[peer] + READY_OFF + fx.Int64(rank * (LINE_WORDS * 4)),
                            epoch + 1,
                        )
                    for peer in range_constexpr(WORLD):
                        _spin(
                            mine + READY_OFF + fx.Int64(peer * (LINE_WORDS * 4)),
                            epoch + 1,
                        )
                    _flag_store(state + 4, 0)
                    _flag_store(state, epoch + 1)

    @flyc.jit
    def launch(
        mla_base: fx.Int64,
        mla_block_stride: fx.Int64,
        mla_pages_per_block: fx.Int32,
        mla_block_table: fx.Int64,
        mla_bt_stride: fx.Int32,
        index_base: fx.Int64,
        index_block_stride: fx.Int64,
        index_block_table: fx.Int64,
        index_bt_stride: fx.Int32,
        seq_lens: fx.Int64,
        query_start_loc: fx.Int64,
        dst_block_table: fx.Int64,
        dst_bt_stride: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        workspace_bytes: fx.Int64,
        dst_index_offset: fx.Int64,
        rank: fx.Int32,
        max_local_pages: fx.Int32,
        num_reqs: fx.Int32,
        state: fx.Int64,
        stream: fx.Stream,
    ):
        grid = num_reqs * max_local_pages * (PAGE // ROWS_PER_CTA)
        kernel(
            mla_base,
            mla_block_stride,
            mla_pages_per_block,
            mla_block_table,
            mla_bt_stride,
            index_base,
            index_block_stride,
            index_block_table,
            index_bt_stride,
            seq_lens,
            query_start_loc,
            dst_block_table,
            dst_bt_stride,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            workspace_bytes,
            dst_index_offset,
            rank,
            max_local_pages,
            num_reqs,
            state,
        ).launch(
            grid=(fx.Int64(grid), 1, 1),
            block=(NTHREADS, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    return launch
