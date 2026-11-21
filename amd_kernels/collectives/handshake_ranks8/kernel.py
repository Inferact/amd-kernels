# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank hand-off kernels: the bare barrier and the ways to move a payload with it.

Protocol. Every rank owns one uncached block mapped on all peers (``common/rank8_comm.py``): ``FLAGS`` bytes of flag
words (workgroup ``b``, source ``s`` at byte ``FLAG_STRIDE b + 4 s``), then a region for the tagged-word exchange and
a region for the data of the other paths, each two parity slots of eight source rows with a row stride that does not
depend on the call's size (calls of different sizes in one stream use the same parity halves; the tagged words never
share a region with payload words, which could equal a tag). A rank raises a flag by an atomic system-scope store
(posted write-through) into the peer's block and polls its own block; the flag word carries the call number (epoch:
the workgroup's counter word + 1, read at entry, written at exit), so nothing is reset and ``(flag - epoch) >= 0``
(signed, wrap-safe) means "arrived". The counter must never be rewound. A call's rows go to slot ``epoch & 1``: a peer
cannot reach call n + 2 before every rank has entered call n + 1, so a slot is never overwritten while it is read. The
8 peer bases are kernel arguments rotated to the rank (``rot[0]`` is its own, ``rot[d]`` rank ``(rank + d) % 8``): no
table load sits on the critical path. Peer ``d`` is served by lane ``d`` (or wave ``d``); atomic stores and loads are
never ``volatile`` (the compiler waits for every volatile access to complete).

``barrier``: lanes 1..7 raise this rank's flag at the seven peers and spin on the seven own flags (``lane``), one wave
per peer (``wave``), or ``pull``: the flag is stored locally and the peers poll it across the link. ``exchange_ll``:
every word travels with the epoch in one 8-byte store and the receiver polls the words themselves (no acknowledgement
wait, no flag, no second read; the tag must match exactly). ``exchange_flag``: payload stores, ``s_waitcnt vmcnt(0)``
for their acknowledgements, a barrier, the flags; the receiver reads the landed rows after the flags.
``exchange_pull``: the rows stay in the sender's block, the receiver polls the sender's flag across the link and reads
its row from there.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.common.rank8_signal import (
    SYS, WORLD, epoch_load, flag_store, peer_lane, raise_flag, sel8, spin_until, wait_flag,
)
from amd_kernels.warm_up import launcher
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

NT = 256
NTF = 512
FLAGS = 4096
FLAG_STRIDE = 128
LL_MAX_WORDS = 1024
FLAG_MAX_WORDS = 16384


LL_ROW = FLAG_MAX_WORDS * 8
DATA_ROW = FLAG_MAX_WORDS * 4
LL_OFF = FLAGS
DATA_OFF = LL_OFF + 2 * WORLD * LL_ROW


def block_bytes():
    return DATA_OFF + 2 * WORLD * DATA_ROW


def _rsrc(addr, nbytes):
    return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


@launcher
def compile_barrier(variant: str, ncta: int = 1):
    nt = 512 if variant in ("wave", "pullw") else 64

    @flyc.kernel(name=f"handshake_ranks8_barrier_{variant}_{ncta}", known_block_size=[nt, 1, 1])
    def kernel(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        rot = [p0, p1, p2, p3, p4, p5, p6, p7]
        cr = _rsrc(arg_ctr, 4096)
        e = epoch_load(cr, bid)
        off = bid * FLAG_STRIDE
        if variant == "lane":
            if peer_lane(tx):
                raise_flag(rot, tx, i32_rank, e, off)
                wait_flag(rot, tx, i32_rank, e, off)
        elif variant == "wave":
            w = tx // 64
            if (tx % 64 == 0) & peer_lane(w):
                raise_flag(rot, w, i32_rank, e, off)
                wait_flag(rot, w, i32_rank, e, off)
            fx.gpu.barrier()
        elif variant == "pullw":
            w = tx // 64
            if tx == 0:
                flag_store(rot[0] + off + i32_rank * 4, e)
            if (tx % 64 == 0) & peer_lane(w):
                spin_until(sel8(w, rot) + (off + ((i32_rank + w) & 7) * 4), e)
            fx.gpu.barrier()
        elif variant == "pull":
            if tx == 0:
                buffer_ops.buffer_store(e, _rsrc(rot[0], FLAGS), bid * (FLAG_STRIDE // 4) + i32_rank,
                                        cache_modifier=SYS)
            if peer_lane(tx):
                spin_until(sel8(tx, rot) + (off + ((i32_rank + tx) & 7) * 4), e)
        if tx == 0:
            buffer_ops.buffer_store(e, cr, bid)

    @flyc.jit
    def launch(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, stream: fx.Stream):
        kernel(arg_ctr, i32_rank, p0, p1, p2, p3, p4, p5, p6, p7).launch(grid=(ncta, 1, 1), block=(nt, 1, 1),
                                                                          stream=stream)

    return launch


@launcher
def compile_exchange_ll(n: int, ncta: int = 1):
    """``out[s] = x_s[me]`` for ``x`` / ``out`` int32 [8, n]; workgroup ``b`` moves words [b n / ncta, (b + 1) n / ncta)
    of every row, n / ncta <= LL_MAX_WORDS."""
    assert n % ncta == 0 and 1 <= n // ncta <= LL_MAX_WORDS and n <= FLAG_MAX_WORDS
    nc = n // ncta
    nt = 64 if nc <= 64 else 128 if nc <= 128 else NT
    k_units = -(-nc // nt)
    row_bytes = n * 8

    @flyc.kernel(name=f"handshake_ranks8_exchange_ll_n{n}_c{ncta}", known_block_size=[nt, 1, 1])
    def kernel(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, arg_x: fx.Int64, arg_out: fx.Int64):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        w0 = bid * nc
        rot = [p0, p1, p2, p3, p4, p5, p6, p7]
        cr = _rsrc(arg_ctr, 4096)
        e = epoch_load(cr, bid)
        par = e & 1
        xr = _rsrc(arg_x, WORLD * n * 4)
        outr = _rsrc(arg_out, WORLD * n * 4)
        send_off = fx.Int64(LL_OFF) + fx.Int64((par * WORLD + i32_rank) * LL_ROW)
        ws = []
        for d in range_constexpr(WORLD):
            for k in range_constexpr(k_units):
                u = tx + k * nt
                ws.append(fx.Int32(buffer_ops.buffer_load(xr, ((i32_rank + d) & 7) * n + w0 + u, vec_width=1,
                                                          dtype=fx.Int32, mask=u < nc)))
        for d in range_constexpr(1, WORLD):
            dr = _rsrc(rot[d] + send_off, row_bytes)
            for k in range_constexpr(k_units):
                u = tx + k * nt
                buffer_ops.buffer_store(Vec.from_elements([ws[d * k_units + k], e], fx.Int32), dr, (w0 + u) * 2,
                                        mask=u < nc, cache_modifier=SYS)
        for k in range_constexpr(k_units):
            u = tx + k * nt
            buffer_ops.buffer_store(ws[k], outr, i32_rank * n + w0 + u, mask=u < nc)
        recv_off = fx.Int64(LL_OFF) + fx.Int64(par * WORLD * LL_ROW)
        recv = [_rsrc(rot[0] + recv_off + fx.Int64(((i32_rank + d) & 7) * LL_ROW), row_bytes)
                for d in range_constexpr(1, WORLD)]

        def poll():
            els = []
            for d in range_constexpr(WORLD - 1):
                for k in range_constexpr(k_units):
                    u = tx + k * nt
                    v = Vec(buffer_ops.buffer_load(recv[d], (w0 + u) * 2, vec_width=2, dtype=fx.Int32, mask=u < nc,
                                                   cache_modifier=SYS))
                    els += [fx.Int32(v[0]), fx.Int32(v[1])]
            return Vec.from_elements(els, fx.Int32)

        def missing(v):
            m = fx.Int32(0)
            for i in range_constexpr((WORLD - 1) * k_units):
                u = tx + (i % k_units) * nt
                m = m | ((u < nc) & (fx.Int32(v[2 * i + 1]) != e)).select(fx.Int32(1), fx.Int32(0))
            return m

        v = poll()
        m = missing(v)
        while m != 0:
            llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
            v = poll()
            m = missing(v)
        for d in range_constexpr(WORLD - 1):
            row = (i32_rank + (d + 1)) & 7
            for k in range_constexpr(k_units):
                u = tx + k * nt
                buffer_ops.buffer_store(fx.Int32(v[2 * (d * k_units + k)]), outr, row * n + w0 + u, mask=u < nc)
        if tx == 0:
            buffer_ops.buffer_store(e, cr, bid)

    @flyc.jit
    def launch(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, arg_x: fx.Int64, arg_out: fx.Int64,
               stream: fx.Stream):
        kernel(arg_ctr, i32_rank, p0, p1, p2, p3, p4, p5, p6, p7, arg_x, arg_out).launch(
            grid=(ncta, 1, 1), block=(nt, 1, 1), stream=stream)

    return launch


@launcher
def compile_exchange_flag(n: int, ncta: int = 1):
    """``out[s] = x_s[me]`` for ``x`` / ``out`` int32 [8, n]; workgroup ``b`` moves words [b n / ncta, (b + 1) n / ncta)
    of every row (n / ncta a multiple of 4)."""
    assert n % ncta == 0 and (n // ncta) % 4 == 0 and n <= FLAG_MAX_WORDS
    nc = n // ncta
    nv = nc // 4
    k_units = -(-nv // NTF)
    row_bytes = n * 4

    @flyc.kernel(name=f"handshake_ranks8_exchange_flag_n{n}_c{ncta}", known_block_size=[NTF, 1, 1])
    def kernel(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, arg_x: fx.Int64, arg_out: fx.Int64):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        w0 = bid * nc
        rot = [p0, p1, p2, p3, p4, p5, p6, p7]
        cr = _rsrc(arg_ctr, 4096)
        e = epoch_load(cr, bid)
        par = e & 1
        xr = _rsrc(arg_x, WORLD * n * 4)
        outr = _rsrc(arg_out, WORLD * n * 4)
        send_off = fx.Int64(DATA_OFF) + fx.Int64((par * WORLD + i32_rank) * DATA_ROW)
        ws = []
        for d in range_constexpr(WORLD):
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                ws.append(Vec(buffer_ops.buffer_load(xr, ((i32_rank + d) & 7) * n + w0 + u * 4, vec_width=4,
                                                     dtype=fx.Int32, mask=u < nv)))
        for d in range_constexpr(1, WORLD):
            dr = _rsrc(rot[d] + send_off, row_bytes)
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                buffer_ops.buffer_store(ws[d * k_units + k], dr, w0 + u * 4, mask=u < nv, cache_modifier=SYS)
        for k in range_constexpr(k_units):
            u = tx + k * NTF
            buffer_ops.buffer_store(ws[k], outr, i32_rank * n + w0 + u * 4, mask=u < nv)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        w = tx // 64
        if (tx % 64 == 0) & peer_lane(w):
            raise_flag(rot, w, i32_rank, e, bid * FLAG_STRIDE)
            wait_flag(rot, w, i32_rank, e, bid * FLAG_STRIDE)
        fx.gpu.barrier()
        recv_off = fx.Int64(DATA_OFF) + fx.Int64(par * WORLD * DATA_ROW)
        ws = []
        for d in range_constexpr(1, WORLD):
            sr = _rsrc(rot[0] + recv_off + fx.Int64(((i32_rank + d) & 7) * DATA_ROW), row_bytes)
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                ws.append(Vec(buffer_ops.buffer_load(sr, w0 + u * 4, vec_width=4, dtype=fx.Int32, mask=u < nv,
                                                     cache_modifier=SYS)))
        for d in range_constexpr(1, WORLD):
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                buffer_ops.buffer_store(ws[(d - 1) * k_units + k], outr, ((i32_rank + d) & 7) * n + w0 + u * 4,
                                        mask=u < nv)
        if tx == 0:
            buffer_ops.buffer_store(e, cr, bid)

    @flyc.jit
    def launch(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, arg_x: fx.Int64, arg_out: fx.Int64,
               stream: fx.Stream):
        kernel(arg_ctr, i32_rank, p0, p1, p2, p3, p4, p5, p6, p7, arg_x, arg_out).launch(
            grid=(ncta, 1, 1), block=(NTF, 1, 1), stream=stream)

    return launch


@launcher
def compile_exchange_pull(n: int):
    """``exchange_flag`` the other way round: a rank writes its rows into its own block and raises its flag there; the
    peers poll that flag across the link and read their row from it. One workgroup."""
    assert n % 4 == 0 and n <= FLAG_MAX_WORDS
    nv = n // 4
    k_units = -(-nv // NTF)
    row_bytes = n * 4

    @flyc.kernel(name=f"handshake_ranks8_exchange_pull_n{n}", known_block_size=[NTF, 1, 1])
    def kernel(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, arg_x: fx.Int64, arg_out: fx.Int64):
        tx = fx.thread_idx.x
        rot = [p0, p1, p2, p3, p4, p5, p6, p7]
        cr = _rsrc(arg_ctr, 4096)
        e = epoch_load(cr, 0)
        par = e & 1
        xr = _rsrc(arg_x, WORLD * n * 4)
        outr = _rsrc(arg_out, WORLD * n * 4)
        rows = fx.Int64(DATA_OFF) + fx.Int64(par * WORLD * DATA_ROW)
        mine = _rsrc(rot[0] + rows, WORLD * DATA_ROW)
        ws = []
        for d in range_constexpr(WORLD):
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                ws.append(Vec(buffer_ops.buffer_load(xr, ((i32_rank + d) & 7) * n + u * 4, vec_width=4,
                                                     dtype=fx.Int32, mask=u < nv)))
        for d in range_constexpr(1, WORLD):
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                buffer_ops.buffer_store(ws[d * k_units + k], mine, ((i32_rank + d) & 7) * (DATA_ROW // 4) + u * 4,
                                        mask=u < nv, cache_modifier=SYS)
        for k in range_constexpr(k_units):
            u = tx + k * NTF
            buffer_ops.buffer_store(ws[k], outr, i32_rank * n + u * 4, mask=u < nv)
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        w = tx // 64
        if tx == 0:
            flag_store(rot[0], e)
        if (tx % 64 == 0) & peer_lane(w):
            spin_until(sel8(w, rot), e)
        fx.gpu.barrier()
        ws = []
        for d in range_constexpr(1, WORLD):
            sr = _rsrc(rot[d] + rows + fx.Int64(i32_rank * DATA_ROW), row_bytes)
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                ws.append(Vec(buffer_ops.buffer_load(sr, u * 4, vec_width=4, dtype=fx.Int32, mask=u < nv,
                                                     cache_modifier=SYS)))
        for d in range_constexpr(1, WORLD):
            for k in range_constexpr(k_units):
                u = tx + k * NTF
                buffer_ops.buffer_store(ws[(d - 1) * k_units + k], outr, ((i32_rank + d) & 7) * n + u * 4,
                                        mask=u < nv)
        if tx == 0:
            buffer_ops.buffer_store(e, cr, 0)

    @flyc.jit
    def launch(arg_ctr: fx.Int64, i32_rank: fx.Int32, p0: fx.Int64, p1: fx.Int64, p2: fx.Int64, p3: fx.Int64,
               p4: fx.Int64, p5: fx.Int64, p6: fx.Int64, p7: fx.Int64, arg_x: fx.Int64, arg_out: fx.Int64,
               stream: fx.Stream):
        kernel(arg_ctr, i32_rank, p0, p1, p2, p3, p4, p5, p6, p7, arg_x, arg_out).launch(
            grid=(1, 1, 1), block=(NTF, 1, 1), stream=stream)

    return launch
