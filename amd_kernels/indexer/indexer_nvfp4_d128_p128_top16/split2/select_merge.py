# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Top-16 of the decode indexer split over two GPUs, after ``decode_score.py``: per (row, head) the top 16 of the
rank's half, exchanged as (key, block) pairs and merged (FlyDSL, gfx950).

Grid (rows, 4): y 0-1 select the peer's heads 2 (1 - me) + y and store the 16 entries into the peer's receive buffer;
y 2-3 select this rank's heads 2 me + y - 2, spin on the peer's entries of the row and merge. An entry is (key, call
count, block, call count), 16 B from one lane: each 8-B half lands whole, so both counts equal to this call's say the
entry is complete (no flag word, no drain before it). The storing workgroups are dispatched first, so the spinning
ones never hold the CUs they need. Selection over the half is the plain four-wave selector
(``topk.top16_indices_p128_plain_wave4``) on the half order (block 2 j + me at j): its ids j go to blocks 2 j + me,
the order preserved. The key is the score's in AITER's order (NaN as -inf); the merged top 16 of the 32 by (key,
block) descending, -1 after the live count, is the one-GPU top 16 bit for bit (the halves are disjoint).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..topk.kernel import _comp, _key, _one
from ..topk.prefill_kernel import INT_MIN, KEY_NEG_INF, KEY_POS_INF, SEL_DWORDS, _rsrc
from ..topk.top16_indices_p128_plain_wave4 import CAND_DWORDS, first_window, row_top16_w4_ids
from .kernel import CNT_MERGE, CNT_SCORE, CNT_WORDS, SC1, SYS


@functools.cache
@launcher
def compile_select_merge():
    name = "index_top16_select_merge_ranks2_p128_w28w64_plain_wave4"

    @fx.struct
    class Shared:
        cand: fx.Array[fx.Int32, CAND_DWORDS, 16]
        sel: fx.Array[fx.Int32, SEL_DWORDS, 16]
        tmp: fx.Array[fx.Int32, 260, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_score: fx.Int64,  # [4, rows, S] fp32, half order
        s_head: fx.Int64,
        s_row: fx.Int64,
        width: fx.Int32,  # positions per score row the selection reads
        arg_seq: fx.Int64,
        ql: fx.Int32,
        arg_ids: fx.Int64,  # scratch [4, rmax, 16] i32: the half's ids
        arg_lrcv: fx.Int64,  # this rank's receive buffer [2, rmax, 16] x (key, count, block, count)
        arg_lpeer: fx.Int64,  # the peer's
        arg_cnt: fx.Int64,
        arg_topk: fx.Int64,  # this rank's heads [2, rows, 16] i32
        t_head: fx.Int64,
        t_row: fx.Int64,
        i32_rmax: fx.Int32,
        i32_me: fx.Int32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        n, y, tx = fx.block_idx.x, fx.block_idx.y, fx.thread_idx.x
        lane = tx % 64
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=CNT_WORDS * 4)
        calls = fx.Int32(
            fx.rocdl.readfirstlane(
                T.i32, fx.Int32(buffer_ops.buffer_load(cr, CNT_MERGE, vec_width=1, dtype=fx.Int32, cache_modifier=SC1))
            )
        )
        if (n == 0) & (y == 0) & (tx == 0):
            buffer_ops.buffer_store(calls, cr, CNT_SCORE, cache_modifier=SYS)
        push = y < 2
        hl = y & 1
        head = push.select(fx.Int32(1) - i32_me, i32_me) * 2 + hl
        row = arg_score + (head.to(fx.Int64) * s_head + n.to(fx.Int64) * s_row) * 4
        pre = first_window(tx, row, width)
        seq = fx.Int32(buffer_ops.buffer_load(
            _rsrc(arg_seq + (n // ql).to(fx.Int64) * 4, 4), fx.Int32(0), vec_width=1, is_scalar=True,
        ))
        kv0 = seq - ql + n % ql + 1
        nblk = ((kv0 > 0).select(kv0, fx.Int32(0)) + 127) // 128
        mine = (nblk > i32_me).select((nblk - i32_me + 1) >> 1, fx.Int32(0))
        ids_row = arg_ids + (head.to(fx.Int64) * i32_rmax.to(fx.Int64) + n.to(fx.Int64)) * 64
        fast = row_top16_w4_ids(tx, row, width, mine, smem.cand.ptr, smem.sel.ptr, smem.tmp.ptr, ids_row, pre)
        j = fx.Int32(smem.tmp.ptr[lane & 15])
        if fast == 0:
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            j = fx.Int32(buffer_ops.buffer_load(_rsrc(ids_row, 64), lane & 15, vec_width=1, dtype=fx.Int32,
                                                cache_modifier=SYS))

        # lanes 0-15: this half's pairs, 16-31: the peer's (own heads); an entry is (key, calls, block, calls), each
        # 8-B half written whole, so the count in both says the entry is this call's
        lr = buffer_ops.create_buffer_resource_from_addr(arg_lrcv, num_records_bytes=fx.Int64(i32_rmax) * 512)
        slot = (hl * i32_rmax + n) * 16 + (lane & 15)
        ok = (lane < 16) & (j >= 0)
        raw = fx.Int32(buffer_ops.buffer_load(_rsrc(row, fx.Int64(width) * 4), j, vec_width=1, dtype=fx.Int32,
                                              mask=ok))
        k = _key(raw)
        k = ((k > KEY_POS_INF) | (k < KEY_NEG_INF)).select(fx.Int32(KEY_NEG_INF), k)
        key = ok.select(k, fx.Int32(INT_MIN))
        blk = ok.select(j * 2 + i32_me, fx.Int32(-1))
        if push:
            if tx < 16:
                lp = buffer_ops.create_buffer_resource_from_addr(arg_lpeer, num_records_bytes=fx.Int64(i32_rmax) * 512)
                buffer_ops.buffer_store(Vec.from_elements([key, calls, blk, calls], fx.Int32), lp, slot * 4,
                                        cache_modifier=SYS)
        else:
            if tx < 64:
                peer = (lane >= 16) & (lane < 32)
                pv = Vec(buffer_ops.buffer_load(lr, slot * 4, vec_width=4, dtype=fx.Int32, mask=peer,
                                                cache_modifier=SYS))
                while fx.Int64(fx.rocdl.ballot(
                        T.i64, peer & ((fx.Int32(pv[1]) != calls) | (fx.Int32(pv[3]) != calls)))) != 0:
                    # the fence keeps the load in the loop
                    _llvm.FenceOp(_llvm.AtomicOrdering.acq_rel, syncscope="workgroup")
                    pv = Vec(buffer_ops.buffer_load(lr, slot * 4, vec_width=4, dtype=fx.Int32, mask=peer,
                                                    cache_modifier=SYS))
                key = peer.select(fx.Int32(pv[0]), key)
                blk = peer.select(fx.Int32(pv[2]), blk)
                live = blk >= 0
                me_c = _comp(key, blk)
                rank = fx.Int32(0)
                for m in range_constexpr(32):
                    km = fx.Int32(fx.rocdl.readlane(T.i32, key, m))
                    bm = fx.Int32(fx.rocdl.readlane(T.i32, blk, m))
                    rank = rank + _one(_comp(km, bm) > me_c)
                smem.tmp.ptr[lane] = fx.Int32(-1)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                if live & (lane < 32) & (rank < 16):
                    smem.tmp.ptr[rank] = blk
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                out = arg_topk + (hl.to(fx.Int64) * t_head + n.to(fx.Int64) * t_row) * 4
                if lane < 16:
                    buffer_ops.buffer_store(fx.Int32(smem.tmp.ptr[lane]), _rsrc(out, 64), lane)

    @flyc.jit
    def launch(
        arg_score: fx.Int64,
        s_head: fx.Int64,
        s_row: fx.Int64,
        width: fx.Int32,
        arg_seq: fx.Int64,
        ql: fx.Int32,
        arg_ids: fx.Int64,
        arg_lrcv: fx.Int64,
        arg_lpeer: fx.Int64,
        arg_cnt: fx.Int64,
        arg_topk: fx.Int64,
        t_head: fx.Int64,
        t_row: fx.Int64,
        i32_rmax: fx.Int32,
        i32_me: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_score, s_head, s_row, width, arg_seq, ql, arg_ids, arg_lrcv, arg_lpeer, arg_cnt, arg_topk, t_head, t_row, i32_rmax, i32_me,
        ).launch(grid=(fx.Int64(rows), 4, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}
    return launch
