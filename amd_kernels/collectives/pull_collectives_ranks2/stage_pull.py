# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Two-rank all-gather and dense reduce-scatter in one launch each: the stage, the publication and the pull overlap
part by part.

Gather (up to six fields): stagers copy each field's input into the own pool slot (counter c -> slot c % 2, the
field's 256 B-aligned offset) and into the own block of the field's output, in chunks of about ``PART_BYTES`` (at
most ``parts`` per field: every chunk costs a hand-off); pullers copy the peer's chunks into the peer's block of the
outputs as soon as they are staged.

Reduce-scatter (bf16 [2 m, H]): stagers copy only the peer's rows of the input into the pool; pullers read the peer's
staged copy of this rank's rows and this rank's own rows of the input, add them in rank order ``(+0 + b0) + b1`` in
fp32 and store the bf16 sum.

Hand-off: after a chunk every stager wave waits for its system-coherent stores (vmcnt 0), the workgroup meets at a
barrier and lane 0 adds 1 to the chunk's monotonic ticket in the uncached ticket buffer the peer maps, without waiting
for the result (a gather field's unused tickets advance by a whole launch, so every ticket counts every launch); a
puller workgroup owns whole chunks (dealt round-robin, so the link stays busy while one waits): its lane 0 spins
until the peer's ticket reaches (call number) x stagers (int32 differences, so the counts may wrap; a communicator
keeps one grid for its lifetime), the workgroup acquires and copies with system-coherent loads. The last workgroup of
the launch stores the call number into the device counter (graph-safe: eager calls and replays interleave). A rank
rewrites slot c % 2 only in call c + 2, after the peer finished its pull of call c (the peer's call c + 1 launch
started after it).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .kernel import NT, SYS, _as_f32, _flag_add, _flag_read, _pk_bf16, _sel2

FIELDS = 6
PULLERS = 16
STAGERS = 64
PARTS = 64  # at most, per field
PART_BYTES = 1 << 19  # a gather field's parts: one per PART_BYTES, 1..PARTS
UNROLL = 4
TICKETS = 0  # byte offset of the chunk tickets in the ticket buffer: field f, part k at TICKETS + 4 (f PARTS + k)
END = 2048
TICKET_BYTES = 4096


def _field_parts(n, parts, part_bytes):
    """A field's parts: one per ``part_bytes``, 1..``parts``."""
    q = n // part_bytes
    return (q < 1).select(fx.Int32(1), (q > parts).select(fx.Int32(parts), q))


def compile_gather_stage_pull(pullers: int = PULLERS, stagers: int = STAGERS, parts: int = PARTS,
                              unroll: int = UNROLL, part_bytes: int = PART_BYTES):
    assert pullers + stagers <= 256 and 4 * FIELDS * parts <= END
    step = NT * 4 * unroll  # dwords per workgroup per loop step
    nwg = pullers + stagers
    name = f"gather6_stage_pull_ranks2_p{pullers}_s{stagers}_c{parts}_u{unroll}_b{part_bytes}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_x0: fx.Int64, arg_x1: fx.Int64, arg_x2: fx.Int64, arg_x3: fx.Int64, arg_x4: fx.Int64, arg_x5: fx.Int64,
        arg_o0: fx.Int64, arg_o1: fx.Int64, arg_o2: fx.Int64, arg_o3: fx.Int64, arg_o4: fx.Int64, arg_o5: fx.Int64,
        off0: fx.Int64, off1: fx.Int64, off2: fx.Int64, off3: fx.Int64, off4: fx.Int64, off5: fx.Int64,
        n0: fx.Int32, n1: fx.Int32, n2: fx.Int32, n3: fx.Int32, n4: fx.Int32, n5: fx.Int32,
        arg_p0: fx.Int64, arg_p1: fx.Int64, arg_t0: fx.Int64, arg_t1: fx.Int64,
        counter: fx.Tensor, i64_stride: fx.Int64, i32_me: fx.Int32,
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        call = fx.Int32(counter[0]) + 1
        slot = fx.Int64((call - 1) & 1) * i64_stride
        peer = 1 - i32_me
        my_t = _sel2(i32_me, arg_t0, arg_t1)
        xs = (arg_x0, arg_x1, arg_x2, arg_x3, arg_x4, arg_x5)
        outs = (arg_o0, arg_o1, arg_o2, arg_o3, arg_o4, arg_o5)
        offs = (off0, off1, off2, off3, off4, off5)
        ns = (n0, n1, n2, n3, n4, n5)
        if bid >= pullers:
            sid = bid - pullers
            pool = _sel2(i32_me, arg_p0, arg_p1) + slot
            for f in range_constexpr(FIELDS):
                n = ns[f]
                n16 = n // 16
                np_ = _field_parts(n, parts, part_bytes)
                src = buffer_ops.create_buffer_resource_from_addr(xs[f], num_records_bytes=n)
                dst = buffer_ops.create_buffer_resource_from_addr(pool + offs[f], num_records_bytes=n)
                own = buffer_ops.create_buffer_resource_from_addr(outs[f] + fx.Int64(i32_me) * fx.Int64(n),
                                                                  num_records_bytes=n)
                k = fx.Int32(0)
                while k < np_:
                    d1 = ((k + 1) * n16 // np_) * 4
                    off = (k * n16 // np_) * 4 + sid * step + tx * 4
                    while off < d1:
                        vals = []
                        for u in range_constexpr(unroll):
                            o = off + u * NT * 4
                            vals.append(Vec(buffer_ops.buffer_load(src, o, vec_width=4, dtype=fx.Int32, mask=o < d1)))
                        for u in range_constexpr(unroll):
                            o = off + u * NT * 4
                            buffer_ops.buffer_store(vals[u], dst, o, mask=o < d1, cache_modifier=SYS)
                            buffer_ops.buffer_store(vals[u], own, o, mask=o < d1)
                        off = off + step * stagers
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    if tx == 0:
                        _flag_add(my_t + TICKETS + fx.Int64((f * parts + k) * 4), 1)
                    k = k + 1
                # every ticket counts every launch: the field's unused parts advance by a whole launch's worth
                if (sid == 0) & (tx == 0):
                    k = np_
                    while k < parts:
                        _flag_add(my_t + TICKETS + fx.Int64((f * parts + k) * 4), stagers)
                        k = k + 1
        else:
            pt = _sel2(peer, arg_t0, arg_t1)
            pool = _sel2(peer, arg_p0, arg_p1) + slot
            base = fx.Int32(0)  # parts of the fields before f: the parts are dealt round-robin over all fields
            for f in range_constexpr(FIELDS):
                n = ns[f]
                n16 = n // 16
                np_ = _field_parts(n, parts, part_bytes)
                src = buffer_ops.create_buffer_resource_from_addr(pool + offs[f], num_records_bytes=n)
                dst = buffer_ops.create_buffer_resource_from_addr(outs[f] + fx.Int64(peer) * fx.Int64(n),
                                                                  num_records_bytes=n)
                k = (bid + pullers * FIELDS * parts - base) % pullers
                while k < np_:
                    d0 = (k * n16 // np_) * 4
                    d1 = ((k + 1) * n16 // np_) * 4
                    if tx == 0:
                        fw = pt + TICKETS + fx.Int64((f * parts + k) * 4)
                        t = _flag_read(fw)
                        while t - call * stagers < 0:
                            t = _flag_read(fw)
                    fx.gpu.barrier()
                    _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
                    off = d0 + tx * 4
                    while off < d1:
                        vals = []
                        for u in range_constexpr(unroll):
                            o = off + u * NT * 4
                            vals.append(Vec(buffer_ops.buffer_load(src, o, vec_width=4, dtype=fx.Int32, mask=o < d1,
                                                                   cache_modifier=SYS)))
                        for u in range_constexpr(unroll):
                            o = off + u * NT * 4
                            buffer_ops.buffer_store(vals[u], dst, o, mask=o < d1)
                        off = off + step
                    k = k + pullers
                base = base + np_
        fx.gpu.barrier()
        if tx == 0:
            if _flag_add(my_t + END, 1) == nwg - 1:
                _flag_add(my_t + END, -nwg)
                counter[0] = call

    @flyc.jit
    def launch_gather_stage_pull(
        arg_x0: fx.Int64, arg_x1: fx.Int64, arg_x2: fx.Int64, arg_x3: fx.Int64, arg_x4: fx.Int64, arg_x5: fx.Int64,
        arg_o0: fx.Int64, arg_o1: fx.Int64, arg_o2: fx.Int64, arg_o3: fx.Int64, arg_o4: fx.Int64, arg_o5: fx.Int64,
        off0: fx.Int64, off1: fx.Int64, off2: fx.Int64, off3: fx.Int64, off4: fx.Int64, off5: fx.Int64,
        n0: fx.Int32, n1: fx.Int32, n2: fx.Int32, n3: fx.Int32, n4: fx.Int32, n5: fx.Int32,
        arg_p0: fx.Int64, arg_p1: fx.Int64, arg_t0: fx.Int64, arg_t1: fx.Int64,
        counter: fx.Tensor, i64_stride: fx.Int64, i32_me: fx.Int32, stream: fx.Stream,
    ):
        kernel(arg_x0, arg_x1, arg_x2, arg_x3, arg_x4, arg_x5, arg_o0, arg_o1, arg_o2, arg_o3, arg_o4, arg_o5,
               off0, off1, off2, off3, off4, off5, n0, n1, n2, n3, n4, n5, arg_p0, arg_p1, arg_t0, arg_t1,
               counter, i64_stride, i32_me).launch(grid=(nwg, 1, 1), block=(NT, 1, 1), stream=stream)

    launch_gather_stage_pull.kernel_name = name
    return launch_gather_stage_pull


def compile_rs_stage_pull(pullers: int = PULLERS, stagers: int = STAGERS, parts: int = PARTS, unroll: int = UNROLL):
    """The dense bf16 reduce-scatter of ``[2 m, H]`` rows (``i64_block`` bytes = m H 2 per rank block): the pool slot
    follows the communicator's ``counter``, the tickets count this kernel's own launches (``calls``)."""
    assert pullers + stagers <= 256 and 4 * parts <= END
    step = NT * 4 * unroll
    nwg = pullers + stagers
    name = f"rs_stage_pull_sum2_ranks2_p{pullers}_s{stagers}_c{parts}_u{unroll}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_x: fx.Int64, arg_out: fx.Int64, arg_p0: fx.Int64, arg_p1: fx.Int64, arg_t0: fx.Int64,
               arg_t1: fx.Int64, counter: fx.Tensor, calls: fx.Tensor, i64_stride: fx.Int64, i64_block: fx.Int64,
               i32_me: fx.Int32):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        seq = fx.Int32(counter[0]) + 1
        slot = fx.Int64((seq - 1) & 1) * i64_stride
        call = fx.Int32(calls[0]) + 1
        peer = 1 - i32_me
        my_t = _sel2(i32_me, arg_t0, arg_t1)
        n16 = fx.Int32(i64_block // 16)
        if bid >= pullers:
            sid = bid - pullers
            src = buffer_ops.create_buffer_resource_from_addr(arg_x + fx.Int64(peer) * i64_block,
                                                              num_records_bytes=i64_block)
            dst = buffer_ops.create_buffer_resource_from_addr(_sel2(i32_me, arg_p0, arg_p1) + slot,
                                                              num_records_bytes=i64_block)
            k = fx.Int32(0)
            while k < parts:
                d1 = ((k + 1) * n16 // parts) * 4
                off = (k * n16 // parts) * 4 + sid * step + tx * 4
                while off < d1:
                    vals = []
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        vals.append(Vec(buffer_ops.buffer_load(src, o, vec_width=4, dtype=fx.Int32, mask=o < d1)))
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        buffer_ops.buffer_store(vals[u], dst, o, mask=o < d1, cache_modifier=SYS)
                    off = off + step * stagers
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx == 0:
                    _flag_add(my_t + TICKETS + fx.Int64(k * 4), 1)
                k = k + 1
        else:
            pt = _sel2(peer, arg_t0, arg_t1)
            rs = buffer_ops.create_buffer_resource_from_addr(_sel2(peer, arg_p0, arg_p1) + slot,
                                                             num_records_bytes=i64_block)
            ls = buffer_ops.create_buffer_resource_from_addr(arg_x + fx.Int64(i32_me) * i64_block,
                                                             num_records_bytes=i64_block)
            out = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i64_block)
            first = i32_me == 0
            k = bid
            while k < parts:
                d0 = (k * n16 // parts) * 4
                d1 = ((k + 1) * n16 // parts) * 4
                if tx == 0:
                    fw = pt + TICKETS + fx.Int64(k * 4)
                    t = _flag_read(fw)
                    while t - call * stagers < 0:
                        t = _flag_read(fw)
                fx.gpu.barrier()
                _llvm.FenceOp(_llvm.AtomicOrdering.acquire, syncscope="")
                off = d0 + tx * 4
                while off < d1:
                    rv = []
                    lv = []
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        rv.append(Vec(buffer_ops.buffer_load(rs, o, vec_width=4, dtype=fx.Int32, mask=o < d1,
                                                             cache_modifier=SYS)))
                        lv.append(Vec(buffer_ops.buffer_load(ls, o, vec_width=4, dtype=fx.Int32, mask=o < d1)))
                    for u in range_constexpr(unroll):
                        o = off + u * NT * 4
                        packed = []
                        for i in range_constexpr(4):
                            a = first.select(fx.Int32(lv[u][i]), fx.Int32(rv[u][i]))
                            b = first.select(fx.Int32(rv[u][i]), fx.Int32(lv[u][i]))
                            lo = (fx.Float32(0.0) + _as_f32(a << 16)) + _as_f32(b << 16)
                            hi = (fx.Float32(0.0) + _as_f32(a & fx.Int32(-65536))) + _as_f32(b & fx.Int32(-65536))
                            packed.append(_pk_bf16(lo, hi))
                        buffer_ops.buffer_store(Vec.from_elements(packed, fx.Int32), out, o, mask=o < d1)
                    off = off + step
                k = k + pullers
        fx.gpu.barrier()
        if tx == 0:
            if _flag_add(my_t + END, 1) == nwg - 1:
                _flag_add(my_t + END, -nwg)
                counter[0] = seq
                calls[0] = call

    @flyc.jit
    def launch_rs_stage_pull(arg_x: fx.Int64, arg_out: fx.Int64, arg_p0: fx.Int64, arg_p1: fx.Int64,
                             arg_t0: fx.Int64, arg_t1: fx.Int64, counter: fx.Tensor, calls: fx.Tensor,
                             i64_stride: fx.Int64, i64_block: fx.Int64, i32_me: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_out, arg_p0, arg_p1, arg_t0, arg_t1, counter, calls, i64_stride, i64_block, i32_me).launch(
            grid=(nwg, 1, 1), block=(NT, 1, 1), stream=stream)

    launch_rs_stage_pull.kernel_name = name
    return launch_rs_stage_pull
