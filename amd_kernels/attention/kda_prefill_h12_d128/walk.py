# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA state walk (gfx950, FlyDSL): the chunks of one (sequence, head) in order, the [K, V] state in MFMA accumulators, publishing
the state each chunk starts from.

Per chunk, with S^T [K, V16] in registers: ``Vn = u - w S^T`` and ``S^T = diag(decay) S^T + kg^T Vn`` (``qg`` and ``Aqk`` are not
read: the output pass computes ``O = qg S^T + Aqk Vn`` from what this kernel publishes). Before the chunk's MFMAs a compute wave writes
the bf16 state fragments the MFMAs consume (``sf``) and, after the first product, ``Vn`` as the bf16 fragments of the state update
(``vnf``): one linear 1 KB piece per (slice, fragment) of the chunk's 48 KB state record ``[S | Vn]`` (write-through stores).

A workgroup has two compute waves (one 16-column value slice each; 4 workgroups per (sequence, head), all on one XCD so the 41 KB
of a chunk's record are fetched from HBM once) and two helper waves. The helpers issue every ``buffer_load ... lds`` and the flag
traffic, so the compute waves issue LDS reads, MFMAs and stores only. A chunk's ``w`` (16 pieces) and ``kgt | decay`` (17 pieces)
move into a ring of four 17 KB slots (W_c, K_c, W_{c+1}, K_{c+1}, ...), its ``u`` slices into a double buffer: slot W_c is refilled
with W_{c+2} once every wave has finished its first product, slot K_c with K_{c+1} when the walk enters chunk c + 1, so three of
the four slots are in flight at any time. Two barriers per chunk (before the first product, before the second) order the DMAs; the
helpers' ``vmcnt`` waits count their own issue order (``sched_barrier`` pins it).

Hand-over with the prologue and the output pass (``flags``): chunk ``c`` is read once the prologue has published ``F1[h, c, sg]``
(DMA'd into LDS three chunks ahead with agent-scope loads and polled there; the single consumer resets it), and the state record
of chunk ``c`` is published by ``F2[h, c, sg]`` once its write-through stores completed (two chunks later in the loop, at the end
for the last two). Without flags the walk runs after the prologue launch (the standalone kernel the tests use).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher

from .common import (
    FLAG_STRIDE,
    WALK_GROUPS,
    barrier,
    lds_load2,
    lds_load4,
    lds_ptr,
    mfma,
    sched_barrier,
    touch_args,
)
from .reference import (
    BT,
    NH,
    RECORD_BYTES,
    RECORD_DEC,
    RECORD_KGT,
    RECORD_U,
    RECORD_W,
    D,
)
from .scan import SC1, STATE_BYTES, _fp, _pack, _u_f32

NSLICE = 8 // WALK_GROUPS  # value slices (16 columns) of a walk workgroup: one per compute wave; the other waves move data
NHELP = 4 - NSLICE
UNIT = 8 * WALK_GROUPS  # a pair's workgroups sit on one XCD: unit u -> pair (u // UNIT) * 8 + u % 8, slice group (u // 8) % WALK_GROUPS
SLOT = 17 * 1024  # a ring slot: w (16 KB) or kgt (16 KB) | decay (1 KB)
U_BASE = 4 * SLOT  # u slices of the compute waves, double buffered: [parity][wave][2 KB]
FLAG_BASE = U_BASE + 2 * NSLICE * 2048  # two 256 B slots of DMA'd flags per helper wave
WALK_LDS = FLAG_BASE + 512 * NHELP
SREC_S = 0
SREC_VN = 32768
SREC_BYTES = 49152  # per (chunk, head): S fragments [8 slices][4][64][16 B], Vn fragments [8 slices][2][64][16 B]
POLL_LIMIT = 1 << 22


def _dma_piece_so(rs, lds_addr, voff, soff):
    d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_addr))
    lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
    _rocdl_ods.raw_ptr_buffer_load_async_lds(
        rs, lp, fx.Int32(16).ir_value(), fx.Int32(voff).ir_value(), fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))


def flags_ints(nc):
    """Int32 flags of ``nc`` chunks: F1 (record ready) then F2 (state record ready) per (head, chunk, slice group)."""
    return 2 * NH * nc * WALK_GROUPS * FLAG_STRIDE


def flag_ready(fr, idx, need):
    """A flag loaded at agent scope (0 when ``need`` is false)."""
    return buffer_ops.buffer_load(fr, idx, vec_width=1, dtype=fx.Int32, mask=need, cache_modifier=SC1)


@ASTRewriter.transform
def spin_until(fr, idx, need, v0, nap=1):
    """Wait until the flag at ``idx`` is non-zero (``v0``: a load already issued), sleeping ``nap`` x 64 cycles between polls; a flag
    that is not ``need``ed is ready."""
    pending = (fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v0))) == 0) & need
    spins = fx.Int32(0)
    while pending & (spins < POLL_LIMIT):
        fx.rocdl.s_sleep(nap)
        v = flag_ready(fr, idx, need)
        pending = (fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v))) == 0) & need
        spins = spins + 1


def dma_flag(rs, lds_addr, idx, need):
    """Flag word ``idx`` into the 256 B LDS slot at ``lds_addr`` (4 bytes per lane, agent scope; 0 when ``need`` is false)."""
    d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_addr))
    lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
    off = need.select(fx.Int32(idx) * 4, fx.Int32(0x7FFFFFF0))
    _rocdl_ods.raw_ptr_buffer_load_async_lds(
        rs, lp, fx.Int32(4).ir_value(), off.ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), SC1))


@ASTRewriter.transform
def spin_lds(rs, lds_addr, idx, need):
    """Wait until the DMA'd flag in the LDS slot is non-zero, polling again with ``vmcnt(0)`` waits; the first read must follow a wait."""
    v = fx.Int32(llvm.load(T.i32, lds_ptr(lds_addr), alignment=4))
    pending = (fx.Int32(fx.rocdl.readfirstlane(T.i32, v)) == 0) & need
    spins = fx.Int32(0)
    while pending & (spins < POLL_LIMIT):
        fx.rocdl.s_sleep(1)
        dma_flag(rs, lds_addr, idx, need)
        fx.rocdl.s_waitcnt(vmcnt=0)
        v = fx.Int32(llvm.load(T.i32, lds_ptr(lds_addr), alignment=4))
        pending = (fx.Int32(fx.rocdl.readfirstlane(T.i32, v)) == 0) & need
        spins = spins + 1


@ASTRewriter.transform
def walk_body(
    lds_base,
    wid,
    ckpt,
    arg_ws,
    i32_nc,
    arg_state,
    i64_slot_bytes,
    arg_sidx,
    arg_has,
    arg_cu,
    arg_choff,
    arg_sb,
    i32_npairs,
    arg_ck,
    i64_ck_row_bytes,
    arg_ckoff,
    arg_ckrow,
    arg_flags,
    arg_ts=None,
):
    """Walk workgroup ``wid``: (pair = (sequence, head), slice group) of the launch; waves 0 and 1 walk a slice each, waves 2 and 3
    issue the DMAs (and the flag traffic) of the next chunk."""
    ngrp = WALK_GROUPS
    nck = 8 if ckpt else 0
    fl = arg_flags is not None
    tx = fx.thread_idx.x
    lane = tx % 64
    wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
    l16 = lane % 16
    qg = lane // 16
    pair = (wid // UNIT) * 8 + wid % 8
    sg = (wid // 8) % ngrp
    if pair < i32_npairs:
        seq = pair // NH
        h = pair % NH
        cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
        tok0 = fx.Int32(buffer_ops.buffer_load(cu_r, seq, vec_width=1, is_scalar=True))
        tok1 = fx.Int32(buffer_ops.buffer_load(cu_r, seq + 1, vec_width=1, is_scalar=True))
        c0 = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_choff), seq * 2, vec_width=1, is_scalar=True))
        slot = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_sidx), seq, vec_width=1, is_scalar=True))
        has_w = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_has), seq // 4, vec_width=1, is_scalar=True))
        if const_expr(ckpt):
            ck_off = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_ckoff), seq, vec_width=1, is_scalar=True))
            ck_row = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_ckrow), seq, vec_width=1, is_scalar=True))
        nch = (tok1 - tok0 + (BT - 1)) // BT
        if nch > 0:
            lane16 = lane * 16
            rec_r = _fp(arg_ws + fx.Int64(h * i32_nc + c0) * RECORD_BYTES, nch * RECORD_BYTES)
            sb_r = _fp(arg_sb + fx.Int64(h * i32_nc + c0) * SREC_BYTES, nch * SREC_BYTES)
            if const_expr(fl):
                f_r = buffer_ops.create_buffer_resource_from_addr(arg_flags, num_records_bytes=fx.Int32(0x7FFFFFF0))
                f2_base = NH * i32_nc * WALK_GROUPS * FLAG_STRIDE

                def f1_idx(c):
                    return ((h * i32_nc + c0 + fx.min(c, nch - 1)) * WALK_GROUPS + sg) * FLAG_STRIDE

                def f2_idx(c):
                    return f2_base + ((h * i32_nc + c0 + c) * WALK_GROUPS + sg) * FLAG_STRIDE

            def so(c):
                return fx.min(c, nch - 1) * RECORD_BYTES

            fx.rocdl.s_setprio(3)
            if wave < NSLICE:
                vs0 = sg * NSLICE + wave
                ubase = lds_base + U_BASE + wave * 2048
                has = ((has_w >> ((seq % 4) * 8)) & 0xFF) != 0
                src_r = _fp(arg_state + fx.Int64(slot) * i64_slot_bytes + fx.Int64(h) * STATE_BYTES, STATE_BYTES)
                s0 = [buffer_ops.buffer_load(src_r, (16 * vs0 + l16) * D + 16 * kt + 4 * qg, vec_width=4, dtype=fx.Float32, mask=has)
                      for kt in range_constexpr(8)]
                sched_barrier()
                for c, st in range(0, fx.Int64(nch), 1, init=[fx.as_ir_value(x) for x in s0]):
                    ci = fx.Int32(c)
                    S = [Vec(st[kt]) for kt in range_constexpr(8)]
                    fx.rocdl.s_waitcnt(vmcnt=6 + nck, lgkmcnt=0)
                    barrier()
                    sched_barrier()
                    if const_expr(arg_ts is not None):
                        t_now = fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])).to(fx.Int32)
                        ts_r = buffer_ops.create_buffer_resource_from_addr(arg_ts, num_records_bytes=fx.Int32(0x7FFFFFF0))
                        if (tx == 0) & (sg == 0):
                            buffer_ops.buffer_store(t_now, ts_r, (1 << 20) + ((h * i32_nc + c0 + ci) * WALK_GROUPS + sg))
                    par = ci % 2
                    u_raw = [lds_load2(ubase + par * (NSLICE * 2048) + mt * 512 + lane * 8) for mt in range_constexpr(4)]
                    wbase = lds_base + par * 2 * SLOT + lane16
                    order = [(mt, ks) for ks in range_constexpr(4) for mt in range_constexpr(4)]
                    wfr = [lds_load4(wbase + (mt * 4 + ks) * 1024) for mt, ks in order]
                    sf = [_pack(S[2 * ks], S[2 * ks + 1]) for ks in range_constexpr(4)]
                    sbc = ci * SREC_BYTES
                    for ks in range_constexpr(4):
                        buffer_ops.buffer_store(sf[ks], sb_r, lane * 4, cache_modifier=SC1, soffset_bytes=sbc + (SREC_S + (vs0 * 4 + ks) * 1024))
                    sched_barrier()
                    vn = _u_f32(u_raw)
                    for i in range_constexpr(16):
                        mt, ks = order[i]
                        vn[mt] = mfma(wfr[i], sf[ks], vn[mt])
                    sched_barrier()
                    vnf = [_pack(vn[2 * ksp], vn[2 * ksp + 1]) for ksp in range_constexpr(2)]
                    for ksp in range_constexpr(2):
                        buffer_ops.buffer_store(vnf[ksp], sb_r, lane * 4, cache_modifier=SC1, soffset_bytes=sbc + (SREC_VN + (vs0 * 2 + ksp) * 1024))
                    fx.rocdl.s_waitcnt(lgkmcnt=0)
                    barrier()
                    sched_barrier()
                    korder = [(kt, ksp) for ksp in range_constexpr(2) for kt in range_constexpr(8)]
                    kbase = lds_base + (par * 2 + 1) * SLOT + lane16
                    kfr = [lds_load4(kbase + (kt * 2 + ksp) * 1024) for kt, ksp in korder]
                    dbase = lds_base + (par * 2 + 1) * SLOT + 16 * 1024 + qg * 16
                    d4s = [lds_load4(dbase + kt * 64).bitcast(fx.Float32) for kt in range_constexpr(8)]
                    s_new = [S[kt] * d4s[kt] for kt in range_constexpr(8)]
                    sched_barrier()
                    for i in range_constexpr(16):
                        kt, ksp = korder[i]
                        s_new[kt] = mfma(kfr[i], vnf[ksp], s_new[kt])
                    sched_barrier()
                    if const_expr(ckpt):
                        hit = ((ci + 1) * BT == ck_off) & (ck_row >= 0) & (ck_off > 0)
                        ck_r = _fp(arg_ck + fx.Int64(fx.max(ck_row, fx.Int32(0))) * i64_ck_row_bytes + fx.Int64(h) * STATE_BYTES, STATE_BYTES)
                        for kt in range_constexpr(8):
                            buffer_ops.buffer_store(s_new[kt], ck_r, (16 * vs0 + l16) * D + 16 * kt + 4 * qg, mask=hit)
                        sched_barrier()
                    results = yield [fx.as_ir_value(x) for x in s_new]
                fx.rocdl.s_waitcnt(vmcnt=0)
                dst_r = _fp(arg_state + fx.Int64(slot) * i64_slot_bytes + fx.Int64(h) * STATE_BYTES, STATE_BYTES)
                for kt in range_constexpr(8):
                    buffer_ops.buffer_store(Vec(results[kt]), dst_r, (16 * vs0 + l16) * D + 16 * kt + 4 * qg)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                barrier()
            else:
                hw = wave - NSLICE
                pl = 1 if fl else 0
                npw = -(-16 // NHELP)
                npk = -(-17 // NHELP)
                npu = -(-2 * NSLICE // NHELP)

                def fsl(k):
                    return lds_base + FLAG_BASE + hw * 512 + (k % 2) * 256

                def dma_w(c):
                    st = (c % 2) * 2
                    for i in range_constexpr(npw):
                        j = fx.min(hw + i * NHELP, fx.Int32(15))
                        _dma_piece_so(rec_r, lds_base + st * SLOT + j * 1024, RECORD_W + j * 1024 + lane16, so(c))
                    sched_barrier()

                def dma_k(c):
                    st = (c % 2) * 2 + 1
                    for i in range_constexpr(npk):
                        j = fx.min(hw + i * NHELP, fx.Int32(16))
                        blk = (j < 16).select(j * 1024 + RECORD_KGT, fx.Int32(RECORD_DEC))
                        _dma_piece_so(rec_r, lds_base + st * SLOT + j * 1024, blk + lane16, so(c))
                    sched_barrier()

                def dma_u(c):
                    for p in range_constexpr(npu):
                        pj = fx.min(hw + NHELP * p, fx.Int32(2 * NSLICE - 1))
                        _dma_piece_so(rec_r, lds_base + U_BASE + (c % 2) * (NSLICE * 2048) + pj * 1024, RECORD_U + 2048 * (sg * NSLICE) + pj * 1024 + lane16, so(c))
                    sched_barrier()

                if const_expr(fl):
                    for c_ in range_constexpr(2):
                        spin_until(f_r, f1_idx(fx.Int32(c_)), fx.Int32(c_) < nch, flag_ready(f_r, f1_idx(fx.Int32(c_)), fx.Int32(c_) < nch))
                    dma_flag(f_r, fsl(fx.Int32(2)), f1_idx(fx.Int32(2)), fx.Int32(2) < nch)
                    sched_barrier()
                dma_w(fx.Int32(0))
                dma_u(fx.Int32(0))
                dma_k(fx.Int32(0))
                dma_w(fx.Int32(1))
                for c in range(fx.Int32(0), nch, fx.Int32(1)):
                    ci = fx.Int32(c)
                    fx.rocdl.s_waitcnt(vmcnt=npk + npw, lgkmcnt=0)
                    barrier()
                    sched_barrier()
                    if const_expr(fl):
                        if (hw == 0) & (lane == 0):
                            buffer_ops.buffer_store(fx.Int32(0), f_r, f1_idx(ci), cache_modifier=SC1)
                        if (hw == 0) & (lane == 0) & (ci >= 2):
                            buffer_ops.buffer_store(fx.Int32(1), f_r, f2_idx(ci - 2), cache_modifier=SC1)
                    if const_expr(fl):
                        dma_flag(f_r, fsl(ci + 3), f1_idx(ci + 3), (ci + 3) < nch)
                        sched_barrier()
                    dma_u(ci + 1)
                    dma_k(ci + 1)
                    fx.rocdl.s_waitcnt(vmcnt=npw + pl + npu + npk, lgkmcnt=0)
                    barrier()
                    sched_barrier()
                    if const_expr(fl):
                        spin_lds(f_r, fsl(ci + 2), f1_idx(ci + 2), (ci + 2) < nch)
                    dma_w(ci + 2)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                barrier()
                if const_expr(fl):
                    if (hw == 0) & (lane == 0) & (nch >= 2):
                        buffer_ops.buffer_store(fx.Int32(1), f_r, f2_idx(nch - 2), cache_modifier=SC1)
                    if (hw == 0) & (lane == 0):
                        buffer_ops.buffer_store(fx.Int32(1), f_r, f2_idx(nch - 1), cache_modifier=SC1)


@launcher
def compile_walk(ckpt: bool = False, ts: bool = False):
    name = "kda_prefill_walk_h12_d128" + ("_ck" if ckpt else "") + ("_ts" if ts else "")

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, WALK_LDS, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_sb: fx.Int64,
        i32_npairs: fx.Int32,
        arg_ck: fx.Int64,
        i64_ck_row_bytes: fx.Int64,
        arg_ckoff: fx.Int64,
        arg_ckrow: fx.Int64,
        arg_ts: fx.Int64,
    ):
        touch_args(arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_sb, i32_npairs, arg_ck,
                   i64_ck_row_bytes, arg_ckoff, arg_ckrow, arg_ts)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        walk_body(fx.Int32(fx.ptrtoint(smem.buf.ptr)), fx.block_idx.x, ckpt, arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has,
                  arg_cu, arg_choff, arg_sb, i32_npairs, arg_ck, i64_ck_row_bytes, arg_ckoff, arg_ckrow, None, arg_ts if ts else None)

    @flyc.jit
    def launch(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_sb: fx.Int64,
        i32_npairs: fx.Int32,
        arg_ck: fx.Int64,
        i64_ck_row_bytes: fx.Int64,
        arg_ckoff: fx.Int64,
        arg_ckrow: fx.Int64,
        arg_ts: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_sb, i32_npairs, arg_ck,
               i64_ck_row_bytes, arg_ckoff, arg_ckrow, arg_ts).launch(grid=(((i32_npairs + 7) // 8) * UNIT, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
