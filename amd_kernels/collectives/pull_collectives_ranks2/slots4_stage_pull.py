# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank pull batch: slot sum + shared add overlapped with the reduce-scatter pull, one launch.

Workgroups 0..pullers-1 pull, the next ``stagers`` stage; pullers + stagers <= 256 so no puller shares a CU with a
stager.

Stagers: the slot-order FP32 sum + shared output of slots4_stage_rows, rounded once to BF16: first the peer's row
block into the own pool slot (counter c -> slot c % 2) in ``parts`` chunks, each published by a ticket and a flag
word (the last of the stagers stores c+1 into it), then the own block straight into its rows of the local staging
``[rows, H]``. Stores are system-coherent (SC0 | SC1, through the L2). After a chunk every wave waits for its stores
(vmcnt 0), the workgroup meets at a barrier and lane 0 takes the chunk's ticket.

Pullers: every one serves the peer. Per part j its lane 0 spins on the peer's flag word j, the workgroup acquires,
then copies its stripes of the part with system-coherent 16-byte loads into the peer's rows of the staging. The last
workgroup of the launch to finish stores c+1 into the device counter.

Flag words: one uncached buffer per rank, mapped by the peer: flags at word j (j < parts), tickets at byte
TICKETS + 4 j, the launch's end ticket at END. A rank rewrites slot c % 2 only in call c+2, after the peer finished
the pull of call c (the peer's call c+1 launch started after it).

``owner``: ``shared`` is [rows / 2, H], this rank's shared expert output of its own rows, added to the own block only
(masked loads: the peer's rows add +0, which only drops a zero's sign; the rank-order sum from +0 drops it anyway).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from amd_kernels.norm.gemma_add_rmsnorm_h6144.add_rmsnorm.common import bf16x8, pk_bf16
from .kernel import SYS, _flag_add, _flag_read, _sel2

H = 6144
NT = 256
UNROLL = 4
PULLERS = 16
STAGERS = 192
PARTS = 16
TICKETS = 256  # byte offsets in the chunk-flag buffer
END = 1024
FLAG_BYTES = 4096


def compile_slots4_stage_pull(pullers: int = PULLERS, stagers: int = STAGERS, parts: int = PARTS,
                              unroll: int = UNROLL, owner: bool = False):
    assert pullers + stagers <= 256 and 4 * parts <= TICKETS
    step = NT * 4 * unroll  # dwords per puller per loop step
    nwg = pullers + stagers
    name = (f"slots4_shared_stage_pull_h6144_ranks2_p{pullers}_s{stagers}_c{parts}_u{unroll}"
            + ("_owner" if owner else ""))

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_slots: fx.Int64, arg_shared: fx.Int64, arg_dst: fx.Int64,
               arg_p0: fx.Int64, arg_p1: fx.Int64, arg_c0: fx.Int64, arg_c1: fx.Int64,
               counter: fx.Tensor, i64_stride: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        i32_seq = fx.Int32(counter[0]) + 1
        slot_off = fx.Int64((i32_seq - 1) & 1) * i64_stride
        br = i32_rows // 2
        peer = 1 - i32_me
        my_c = _sel2(i32_me, arg_c0, arg_c1)
        if bid >= pullers:
            sid = bid - pullers
            pool = _sel2(i32_me, arg_p0, arg_p1) + slot_off
            hr = buffer_ops.create_buffer_resource_from_addr(
                arg_shared, num_records_bytes=fx.Int64(br if owner else i32_rows) * H * 2)
            k = fx.Int32(0)
            while k < parts + 1:
                # k < parts: part k of the peer's block; k == parts: the whole own block
                last = k == parts
                r0 = last.select(i32_me * br, peer * br + k * br // parts)
                r1 = last.select((i32_me + 1) * br, peer * br + (k + 1) * br // parts)
                dst = buffer_ops.create_buffer_resource_from_addr(
                    last.select(arg_dst, pool), num_records_bytes=fx.Int64(i32_rows) * H * 2)
                piece = sid
                while piece < (r1 - r0) * 3:
                    row = r0 + piece // 3
                    col = (piece % 3) * (NT * 4) + tx * 4
                    off = row * (H // 2) + col
                    sr = buffer_ops.create_buffer_resource_from_addr(
                        arg_slots + fx.Int64(row) * (4 * H * 2), num_records_bytes=4 * H * 2)
                    vals = [bf16x8(Vec(buffer_ops.buffer_load(sr, col + q * (H // 2), vec_width=4, dtype=fx.Int32)))
                            for q in range_constexpr(4)]
                    if const_expr(owner):
                        shared = bf16x8(Vec(buffer_ops.buffer_load(hr, off - i32_me * br * (H // 2), vec_width=4,
                                                                   dtype=fx.Int32, mask=last)))
                    else:
                        shared = bf16x8(Vec(buffer_ops.buffer_load(hr, off, vec_width=4, dtype=fx.Int32)))
                    acc = [(((vals[0][e] + vals[1][e]) + vals[2][e]) + vals[3][e]) + shared[e]
                           for e in range_constexpr(8)]
                    packed = [pk_bf16(acc[2 * e], acc[2 * e + 1]) for e in range_constexpr(4)]
                    buffer_ops.buffer_store(Vec.from_elements(packed, fx.Int32), dst, off, cache_modifier=SYS)
                    piece = piece + stagers
                if k < parts:
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    if tx == 0:
                        tk = my_c + TICKETS + fx.Int64(k * 4)
                        if _flag_add(tk, 1) == stagers - 1:
                            _flag_add(tk, -stagers)
                            fr = buffer_ops.create_buffer_resource_from_addr(my_c, num_records_bytes=TICKETS)
                            buffer_ops.buffer_store(i32_seq, fr, k, cache_modifier=SYS)
                k = k + 1
        else:
            fa = _sel2(peer, arg_c0, arg_c1)
            block = fx.Int64(br) * (H * 2)
            srs = buffer_ops.create_buffer_resource_from_addr(
                _sel2(peer, arg_p0, arg_p1) + slot_off + fx.Int64(i32_me * br) * (H * 2), num_records_bytes=block)
            drs = buffer_ops.create_buffer_resource_from_addr(arg_dst + fx.Int64(peer) * block, num_records_bytes=block)
            j = fx.Int32(0)
            while j < parts:
                if tx == 0:
                    fw = fa + fx.Int64(j * 4)
                    f = _flag_read(fw)
                    while f < i32_seq:
                        f = _flag_read(fw)
                fx.gpu.barrier()
                _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
                d1 = ((j + 1) * br // parts) * (H // 2)
                off = (j * br // parts) * (H // 2) + bid * step + tx * 4
                while off < d1:
                    vals = []
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        vals.append(Vec(buffer_ops.buffer_load(srs, o, vec_width=4, dtype=fx.Int32, mask=o < d1,
                                                               cache_modifier=SYS)))
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        buffer_ops.buffer_store(vals[u], drs, o, mask=o < d1)
                    off = off + step * pullers
                j = j + 1
        fx.gpu.barrier()
        if tx == 0:
            if _flag_add(my_c + END, 1) == nwg - 1:
                _flag_add(my_c + END, -nwg)
                counter[0] = i32_seq

    @flyc.jit
    def launch_slots4_stage_pull(
            arg_slots: fx.Int64, arg_shared: fx.Int64, arg_dst: fx.Int64,
            arg_p0: fx.Int64, arg_p1: fx.Int64, arg_c0: fx.Int64, arg_c1: fx.Int64,
            counter: fx.Tensor, i64_stride: fx.Int64, i32_rows: fx.Int32, i32_me: fx.Int32, stream: fx.Stream):
        kernel(arg_slots, arg_shared, arg_dst, arg_p0, arg_p1, arg_c0, arg_c1, counter, i64_stride, i32_rows,
               i32_me).launch(grid=(nwg, 1, 1), block=(NT, 1, 1), stream=stream)

    launch_slots4_stage_pull.kernel_name = name
    return launch_slots4_stage_pull
