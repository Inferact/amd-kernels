# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunk scan (gfx950, FlyDSL): the chunks of a (sequence, head) in order, the [K, V] state in MFMA accumulators, one
wave per 16 value columns.

Per chunk, with S^T [K, V16] in registers (``Vn = u - w S^T; O = qg S^T + Aqk Vn; S^T = diag(decay) S^T + kg^T Vn``):
every MFMA takes a workspace fragment (reference.py, fragment-major: one fragment is one 1 KB linear read) as one operand and
a register-resident operand as the other. ``Vn`` leaves the first GEMM as accumulators (lane (v, q) holds tokens 4 q + e of tile mt) and
is the B operand of the state update and the A operand of the Aqk product with no data movement; the state accumulators (lane
(v, q) holds k = 16 kt + 4 q + e) feed the next chunk's GEMM the same way, the reduction order being permuted identically on
both operands (columns 32 ks + 16 (s // 4) + 4 q + s % 4 at slot s). O is computed transposed (O^T = S qg^T), so a lane
holds four consecutive value columns of one token and stores 8 bytes.

Data path: a chunk's record (w | qg | kgt | aqk | decay) moves HBM / L2 -> LDS with asynchronous ``buffer_load ... lds``
(linear 1 KB pieces split over the waves, issued between the chunk's MFMAs) into a two-stage ring: before chunk c every wave
waits for its own DMAs and the workgroup barriers, which also proves the other stage free, then the DMAs of chunk c + 1 are
issued. The ``u`` slices of a wave go straight to registers one chunk ahead; a chunk's output tiles are stored at the start
of the next one (rows past the sequence end fall out of the output buffer's range).

Long sequences are cut into G groups of chunks walked in parallel (exact, ``S_out = S_in M_g + B_g`` with the group's transfer
``M_g = prod_c (diag(decay_c) - w_c^T kg_c)``). The ``chain`` kernel walks a group from a zero state with ``u`` (the 8 value
slices of ``B_g``) and from the identity with ``u = 0`` (8 more slices: the rows of ``M_g``, kept as ``diag(A_g)`` in fp32,
``A_g`` the group's decay product, plus ``C_g = M_g - diag(A_g)`` in bf16 fragments); it also marks every row of the
``S_in`` buffer unwritten. The ``walk`` kernel (``foldwalk``) starts with the fold workgroups, one per (pair, value slice): wave 0
composes ``S_in(g + 1) = S_in(g) A_g + S_in(g) C_g + B_g`` over the groups with write-through stores, the other waves only issue
its DMAs. The walk workgroups of group g poll the rows ``S_in(g)`` (agent-scope loads) until none holds the marker, so a group
starts as soon as the fold reaches it and nothing depends on a flag or a fence; the workgroups are dispatched fold first, so
every poll waits on workgroups that are already running. A sequence of 4 chunks or fewer is one group: the walk alone, from the
state cache.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import barrier, bf16x2_hi, bf16x2_lo, lds_load2, lds_load4, lds_store_i16, mfma, sched_barrier, touch_args
from .reference import (BT, D, NH, RECORD_AQK, RECORD_BYTES, RECORD_DEC, RECORD_KGT, RECORD_QG, RECORD_U,
                        RECORD_W)

STATE_BYTES = D * D * 4
SENTINEL = 0x7FC0DEAD  # a NaN payload no state holds: marks a published row not yet written
SC1 = 0x10  # agent-scope (cross-XCD) stores and loads
POLL_LIMIT = 1 << 22  # polls of a published row before the walk gives up
WALK_PIECES = RECORD_U // 1024  # 57 1 KB pieces of w | qg | kgt | aqk | decay
WALK_STAGE = 60 * 1024
CHAIN_PIECES = 33  # w | kgt | decay
CHAIN_STAGE = 34 * 1024
CHAIN_KGT, CHAIN_DEC = 16384, 32768
FRAGS_C = 32768  # bytes of the C_g fragments of one (sequence, head, group)


def _fp(addr, nbytes):
    return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def _group(nt, g_cap, min_chunks):
    """(groups, first chunk, chunk count) helpers of a sequence of ``nt`` chunks."""
    return fx.max(fx.min(nt // min_chunks, g_cap), fx.Int32(1))


def _first(nt, gn, g):
    per = nt // gn
    rem = nt - per * gn
    return g * per + fx.min(g, rem)


def _count(nt, gn, g):
    per = nt // gn
    rem = nt - per * gn
    return per + (g < rem).select(fx.Int32(1), fx.Int32(0))


def _dma_piece(rs, lds_addr, voff):
    d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_addr))
    lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
    _rocdl_ods.raw_ptr_buffer_load_async_lds(
        rs, lp, fx.Int32(16).ir_value(), fx.Int32(voff).ir_value(), fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
        aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))


def _pack(lo, hi):
    return Vec.from_elements([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def _u_f32(raw):
    out = []
    for x in raw:
        a, b = fx.Int32(Vec(x)[0]), fx.Int32(Vec(x)[1])
        out.append(Vec.from_elements([bf16x2_lo(a), bf16x2_hi(a), bf16x2_lo(b), bf16x2_hi(b)], fx.Float32))
    return out


def _block_decode(bid, groups, ngrp):
    """(pair, group, slice group) of a block. Workgroup b runs on XCD b % 8; the (pair, group) units are dealt to the XCDs
    round-robin and the ``ngrp`` slice groups of a unit (which share one L2's worth of workspace) follow each other on one XCD."""
    x = bid % 8
    j = bid // 8
    unit = (j // ngrp) * 8 + x
    return unit // groups, unit % groups, j % ngrp


@launcher
def compile_scan(waves: int, nvt: int, chain: bool, stages: int = 2, ckpt: bool = False, skeleton: bool = False,
                 foldwalk: bool = False):
    """``chain=False``: the walk kernel; ``chain=True``: the transfer kernel; ``stages``: depth of the LDS ring (the DMAs and the
    ``u`` loads of chunk c + stages - 1 are issued during chunk c); ``ckpt``: the walk also stores the state after
    ``checkpoint_offsets[n]`` tokens of a sequence into ``checkpoint_state``; ``skeleton``: the same loads, DMAs and stores with
    no math (the benchmark's empirical floor); ``foldwalk``: the walk launch also hosts the fold (the first ``nbfold`` workgroups
    are fold workgroups of one wave): a walk workgroup of group g reads the state S_in(g) the fold publishes, polling for it."""
    nsl = waves * nvt
    nslices = 16 if chain else 8
    ngrp = nslices // nsl
    assert nslices % nsl == 0
    nthreads = 64 * waves
    pieces = CHAIN_PIECES if chain else WALK_PIECES
    stage = CHAIN_STAGE if chain else WALK_STAGE
    ndma = -(-pieces // waves)
    ct_pitch = 16 * nsl + 4  # elements of a staged C^T row
    name = f"kda_prefill_{'chain' if chain else 'walk'}_h12_d128_w{waves}_v{nvt}_s{stages}" + ("_ck" if ckpt else "") + ("_skel" if skeleton else "") + ("_fw" if foldwalk else "")
    fold_stages = 3
    fold_stage = FRAGS_C + 1024 + 8192
    pieces_f = 32 + 1 + 8
    lds_bytes = max(stages * stage, fold_stages * fold_stage) if foldwalk else stages * stage

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=name, known_block_size=[nthreads, 1, 1])
    def kernel(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        arg_gb: fx.Int64,
        arg_gc: fx.Int64,
        arg_ga: fx.Int64,
        arg_sin: fx.Int64,
        i32_groups: fx.Int32,
        i32_minc: fx.Int32,
        i32_npairs: fx.Int32,
        arg_ck: fx.Int64,
        i64_ck_row_bytes: fx.Int64,
        arg_ckoff: fx.Int64,
        arg_ckrow: fx.Int64,
        i32_nbfold: fx.Int32,
    ):
        touch_args(arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_out, i32_ostride, arg_gb, arg_gc,
                   arg_ga, arg_sin, i32_groups, i32_minc, i32_npairs, arg_ck, i64_ck_row_bytes, arg_ckoff, arg_ckrow, i32_nbfold)
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16 = lane % 16
        qg = lane // 16
        if const_expr(foldwalk):
            def fold_role(fb):
                """One workgroup per (pair, V slice): wave 0 computes the rows of S_in(g + 1) = S_in(g) A_g + S_in(g) C_g + B_g for
                every group and publishes them with write-through stores (the walk workgroups poll for them); the other waves only
                issue the DMAs of the next stages."""
                fpair, _fg, fsg = _block_decode(fb, 1, 8)
                if fpair < i32_npairs:
                    n = fpair // NH
                    h = fpair % NH
                    cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
                    tok0 = fx.Int32(buffer_ops.buffer_load(cu_r, n, vec_width=1, is_scalar=True))
                    tok1 = fx.Int32(buffer_ops.buffer_load(cu_r, n + 1, vec_width=1, is_scalar=True))
                    slot = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_sidx), n, vec_width=1, is_scalar=True))
                    has_w = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_has), n // 4, vec_width=1, is_scalar=True))
                    nt = (tok1 - tok0 + (BT - 1)) // BT
                    gn = _group(nt, i32_groups, i32_minc)
                    if gn > 1:
                        has_b = ((has_w >> ((n % 4) * 8)) & 0xFF) != 0
                        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
                        lane16 = lane * 16
                        nh = fpair * i32_groups
                        steps = gn - 1
                        nhelp = nthreads // 64 - 1
                        per_help = -(-pieces_f // nhelp)

                        def fdma(gp, st):
                            gpc = fx.min(gp, steps - 1)
                            rc = _fp(arg_gc + fx.Int64(nh + gpc) * FRAGS_C, FRAGS_C)
                            ra = _fp(arg_ga + fx.Int64(nh + gpc) * (D * 4), 1024)
                            rb = _fp(arg_gb + fx.Int64(nh + gpc) * STATE_BYTES + fx.Int64(fsg * 8192), 8192)
                            for i in range_constexpr(per_help):
                                j = fx.min(wave - 1 + i * nhelp, fx.Int32(pieces_f - 1))
                                dst = lds_base + st * fold_stage + j * 1024
                                if j < 32:
                                    _dma_piece(rc, dst, j * 1024 + lane16)
                                elif j < 33:
                                    _dma_piece(ra, dst, lane16)
                                else:
                                    _dma_piece(rb, dst, (j - 33) * 1024 + lane16)

                        if wave == 0:
                            src_r = _fp(arg_state + fx.Int64(slot) * i64_slot_bytes + fx.Int64(h) * STATE_BYTES, STATE_BYTES)
                            s0f = [buffer_ops.buffer_load(src_r, (16 * fsg + l16) * D + 16 * kt + 4 * qg, vec_width=4, dtype=fx.Float32, mask=has_b)
                                   for kt in range_constexpr(8)]
                            sin0 = _fp(arg_sin + fx.Int64(nh) * STATE_BYTES, STATE_BYTES)
                            for kt in range_constexpr(8):
                                buffer_ops.buffer_store(s0f[kt], sin0, (16 * fsg + l16) * D + 16 * kt + 4 * qg, cache_modifier=SC1)
                            for gp, st in range(0, fx.Int64(steps), 1, init=[fx.as_ir_value(x) for x in s0f]):
                                gi = fx.Int32(gp)
                                S = [Vec(st[kt]) for kt in range_constexpr(8)]
                                barrier()
                                base = lds_base + (gi % fold_stages) * fold_stage
                                a4 = [lds_load4(base + FRAGS_C + (16 * kt + 4 * qg) * 4).bitcast(fx.Float32) for kt in range_constexpr(8)]
                                acc = [S[kt] * a4[kt] + lds_load4(base + FRAGS_C + 1024 + (l16 * D + 16 * kt + 4 * qg) * 4).bitcast(fx.Float32)
                                       for kt in range_constexpr(8)]
                                sf = [_pack(S[2 * ks], S[2 * ks + 1]) for ks in range_constexpr(4)]
                                for ks in range_constexpr(0 if skeleton else 4):
                                    for kt in range_constexpr(8):
                                        acc[kt] = mfma(lds_load4(base + (kt * 4 + ks) * 1024 + lane16), sf[ks], acc[kt])
                                dst = _fp(arg_sin + fx.Int64(nh + gi + 1) * STATE_BYTES, STATE_BYTES)
                                for kt in range_constexpr(8):
                                    buffer_ops.buffer_store(acc[kt], dst, (16 * fsg + l16) * D + 16 * kt + 4 * qg, cache_modifier=SC1)
                                results = yield [fx.as_ir_value(x) for x in acc]
                        else:
                            for p_ in range_constexpr(fold_stages - 1):
                                fdma(fx.Int32(p_), fx.Int32(p_))
                            for gp in range(fx.Int32(0), steps, fx.Int32(1)):
                                gi = fx.Int32(gp)
                                fx.rocdl.s_waitcnt(vmcnt=(fold_stages - 2) * per_help)
                                barrier()
                                fdma(gi + fold_stages - 1, (gi + fold_stages - 1) % fold_stages)

        if const_expr(chain):
            # the fold's inputs are published by sentinel: every state row of the launch starts as an unwritten marker
            sin_units = i32_npairs * i32_groups * (STATE_BYTES // 16)
            sin_r = _fp(arg_sin, i32_npairs * i32_groups * STATE_BYTES)
            sent = Vec.filled(4, SENTINEL, fx.Int32)
            nthr_all = fx.grid_dim.x * nthreads
            for u0_ in range(bid * nthreads + tx, sin_units, nthr_all):
                buffer_ops.buffer_store(sent, sin_r, u0_ * 4)
        if const_expr(foldwalk):
            wbid = bid - i32_nbfold
            fold_blk = bid < i32_nbfold
        else:
            wbid = bid
        pair, g, sg = _block_decode(wbid, i32_groups, ngrp)
        if const_expr(foldwalk):
            if fold_blk:
                fold_role(bid)
        active = (pair < i32_npairs) & (bid >= i32_nbfold) if foldwalk else pair < i32_npairs
        if active:
            n = pair // NH
            h = pair % NH
            cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
            tok0 = fx.Int32(buffer_ops.buffer_load(cu_r, n, vec_width=1, is_scalar=True))
            tok1 = fx.Int32(buffer_ops.buffer_load(cu_r, n + 1, vec_width=1, is_scalar=True))
            c0 = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_choff), n * 2, vec_width=1, is_scalar=True))
            slot = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_sidx), n, vec_width=1, is_scalar=True))
            has_w = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_has), n // 4, vec_width=1, is_scalar=True))
            if const_expr(ckpt):
                ck_off = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_ckoff), n, vec_width=1, is_scalar=True))
                ck_row = fx.Int32(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_ckrow), n, vec_width=1, is_scalar=True))
            ntok = tok1 - tok0
            nt = (ntok + (BT - 1)) // BT
            gn = _group(nt, i32_groups, i32_minc)
            live = (g < gn - 1) if chain else (g < gn)
            if live & (nt > 0):
                lo = _first(nt, gn, g)
                nch = _count(nt, gn, g)
                nhg = pair * i32_groups + g
                lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
                lane16 = lane * 16
                vs0 = (sg * waves + wave) * nvt
                rec0 = arg_ws + fx.Int64(h * i32_nc + c0 + lo) * RECORD_BYTES

                def rec_addr(c):
                    return rec0 + fx.Int64(fx.min(c, nch - 1)) * RECORD_BYTES

                # ---- initial state
                if const_expr(chain):
                    real = vs0 < 8
                    xs = vs0 - 8
                    s0 = []
                    for vt in range_constexpr(nvt):
                        for kt in range_constexpr(8):
                            vals = []
                            for e in range_constexpr(4):
                                hit = (xs + vt == kt) & (4 * qg + e == l16)
                                vals.append(hit.select(fx.Float32(1.0), fx.Float32(0.0)))
                            s0.append(Vec.from_elements(vals, fx.Float32))
                else:
                    has = ((has_w >> ((n % 4) * 8)) & 0xFF) != 0
                    grouped = gn > 1
                    src = grouped.select(fx.Int64(arg_sin) + fx.Int64(nhg) * STATE_BYTES, fx.Int64(arg_state) + fx.Int64(slot) * i64_slot_bytes + fx.Int64(h) * STATE_BYTES)
                    ok = grouped | has
                    src_r = _fp(src, STATE_BYTES)

                    def rd_s0():
                        return [buffer_ops.buffer_load(src_r, (16 * (vs0 + vt) + l16) * D + 16 * kt + 4 * qg, vec_width=4, dtype=fx.Float32,
                                                       mask=ok, cache_modifier=SC1 if foldwalk else 0)
                                for vt in range_constexpr(nvt) for kt in range_constexpr(8)]

                    def unwritten(vs):
                        hit = vs[0][0].bitcast(fx.Int32) == SENTINEL
                        for v_ in vs:
                            for e_ in range_constexpr(4):
                                hit = hit | (v_[e_].bitcast(fx.Int32) == SENTINEL)
                        return fx.Int64(fx.rocdl.ballot(T.i64, hit)) != fx.Int64(0)

                if const_expr(not chain):
                    out_r = buffer_ops.create_buffer_resource_from_addr(
                        arg_out + fx.Int64(tok0 + lo * BT) * (i32_ostride * 2), num_records_bytes=(ntok - lo * BT) * (i32_ostride * 2))
                    out_lane = ((l16 * i32_ostride + h * D + 16 * vs0 + 4 * qg) * 2) >> 2

                def dma(c, st):
                    rs = _fp(rec_addr(c), RECORD_BYTES)
                    jobs = []
                    for i in range_constexpr(ndma):
                        def job(i=i):
                            j = fx.min(wave + i * waves, fx.Int32(pieces - 1))
                            if const_expr(chain):
                                blk = (j < 16).select(j, (j < 32).select(j + 16, fx.Int32(RECORD_DEC // 1024)))
                            else:
                                blk = j
                            _dma_piece(rs, lds_base + st * stage + j * 1024, blk * 1024 + lane16)
                        jobs.append(job)
                    return jobs

                def load_u(c):
                    out = []
                    for vt in range_constexpr(nvt):
                        rs = _fp(rec_addr(c) + (RECORD_U + 2048 * (vs0 + vt)), 2048)
                        for mt in range_constexpr(4):
                            if const_expr(chain):
                                out.append(buffer_ops.buffer_load(rs, (mt * 64 + lane) * 2, vec_width=2, dtype=fx.Int32, mask=real))
                            else:
                                out.append(buffer_ops.buffer_load(rs, (mt * 64 + lane) * 2, vec_width=2, dtype=fx.Int32))
                    return out

                win0 = []
                for p in range_constexpr(stages - 1):
                    raw_p = load_u(fx.Int32(p))
                    for jb in dma(fx.Int32(p), fx.Int32(p)):
                        jb()
                    win0.append(raw_p)
                if const_expr(not chain):
                    s0 = rd_s0()
                    if const_expr(foldwalk):
                        pending = unwritten(s0)
                        spins = fx.Int32(0)
                        while pending & (spins < POLL_LIMIT):
                            fx.rocdl.s_sleep(2)
                            s0 = rd_s0()
                            pending = unwritten(s0)
                            spins = spins + 1
                u0 = _u_f32(win0[0])
                extra0 = ([Vec.filled(4, 1.0, fx.Float32) for _ in range_constexpr(8)] if chain
                          else [Vec.filled(2, 0, fx.Int32) for _ in range_constexpr(nvt * 4)])
                nu = 4 * nvt
                init = [fx.as_ir_value(x) for x in s0 + u0 + extra0 + [y for w in win0[1:] for y in w]]
                n_extra = len(extra0)
                for c, st in range(0, fx.Int64(nch), 1, init=init):
                    ci = fx.Int32(c)
                    S = [[Vec(st[vt * 8 + kt]) for kt in range_constexpr(8)] for vt in range_constexpr(nvt)]
                    U = [[Vec(st[nvt * 8 + vt * 4 + mt]) for mt in range_constexpr(4)] for vt in range_constexpr(nvt)]
                    EX = [Vec(st[nvt * 12 + i]) for i in range_constexpr(n_extra)]
                    win = [[st[nvt * 12 + n_extra + w * nu + i] for i in range_constexpr(nu)] for w in range_constexpr(stages - 2)]
                    fx.rocdl.s_waitcnt(vmcnt=(stages - 2) * (nu + ndma), lgkmcnt=0)
                    barrier()
                    base = lds_base + (ci % stages) * stage + lane16
                    if const_expr(not chain):
                        prow = ((ci - 1) * BT * (i32_ostride * 2)) >> 2
                        for vt in range_constexpr(nvt):
                            for nt_ in range_constexpr(4):
                                buffer_ops.buffer_store(EX[vt * 4 + nt_], out_r, out_lane + prow + ((nt_ * 16 * (i32_ostride * 2)) >> 2) + 8 * vt)
                    new_raw = load_u(ci + stages - 1)
                    jobs = dma(ci + stages - 1, (ci + stages - 1) % stages)
                    sched_barrier()
                    tasks = []
                    if const_expr(chain):
                        for ks in range_constexpr(4):
                            for mt in range_constexpr(4):
                                tasks.append(("W", mt, ks, (mt * 4 + ks) * 1024))
                        tasks.append(("V", 0, 0, 0))
                        for ksp in range_constexpr(2):
                            for kt in range_constexpr(8):
                                tasks.append(("K", kt, ksp, CHAIN_KGT + (kt * 2 + ksp) * 1024))
                        dec_off = CHAIN_DEC
                    else:
                        for ks in range_constexpr(4):
                            for mt in range_constexpr(4):
                                tasks.append(("W", mt, ks, RECORD_W + (mt * 4 + ks) * 1024))
                            for nt_ in range_constexpr(4):
                                tasks.append(("Q", nt_, ks, RECORD_QG + (nt_ * 4 + ks) * 1024))
                        tasks.append(("V", 0, 0, 0))
                        for ksp in range_constexpr(2):
                            for nt_ in range_constexpr(2 * ksp, 4):
                                tasks.append(("A", nt_, ksp, RECORD_AQK + (nt_ * 2 + ksp) * 1024))
                        for ksp in range_constexpr(2):
                            for kt in range_constexpr(8):
                                tasks.append(("K", kt, ksp, RECORD_KGT + (kt * 2 + ksp) * 1024))
                        dec_off = RECORD_DEC
                    if const_expr(skeleton):
                        tasks = []
                    reads = [i for i, t in enumerate(tasks) if t[0] != "V"]
                    frags = {}

                    def rd(r):
                        i = reads[r]
                        frags[i] = lds_load4(base + tasks[i][3])

                    vn = [[U[vt][mt] for mt in range_constexpr(4)] for vt in range_constexpr(nvt)]
                    ob = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)] for _ in range_constexpr(nvt)]
                    sf = {}
                    s_new = [[S[vt][kt] for kt in range_constexpr(8)] if skeleton else [None] * 8 for vt in range_constexpr(nvt)]
                    vnf = []
                    d4s = None
                    nread = len(reads)
                    depth = 8
                    for r in range_constexpr(min(depth, nread)):
                        rd(r)
                    done = 0
                    for i, tk in enumerate(tasks):
                        kind = tk[0]
                        if kind != "V":
                            r = done
                            done += 1
                            if r + depth < nread:
                                rd(r + depth)
                            if r == 2:
                                d4s = [lds_load4(base - lane16 + dec_off + kt * 64 + qg * 16).bitcast(fx.Float32) for kt in range_constexpr(8)]
                            f = frags[i]
                            if kind == "W" and tk[1] == 0:
                                for vt in range_constexpr(nvt):
                                    sf[(vt, tk[2])] = _pack(S[vt][2 * tk[2]], S[vt][2 * tk[2] + 1])
                            if kind == "W":
                                for vt in range_constexpr(nvt):
                                    vn[vt][tk[1]] = mfma(f, sf[(vt, tk[2])], vn[vt][tk[1]])
                            elif kind == "Q":
                                for vt in range_constexpr(nvt):
                                    ob[vt][tk[1]] = mfma(sf[(vt, tk[2])], f, ob[vt][tk[1]])
                            elif kind == "A":
                                for vt in range_constexpr(nvt):
                                    ob[vt][tk[1]] = mfma(vnf[vt][tk[2]], f, ob[vt][tk[1]])
                            else:
                                for vt in range_constexpr(nvt):
                                    s_new[vt][tk[1]] = mfma(f, vnf[vt][tk[2]], s_new[vt][tk[1]])
                        else:
                            for kt in range_constexpr(8):
                                for vt in range_constexpr(nvt):
                                    s_new[vt][kt] = S[vt][kt] * d4s[kt]
                            for vt in range_constexpr(nvt):
                                vnf.append([_pack(vn[vt][2 * ksp], vn[vt][2 * ksp + 1]) for ksp in range_constexpr(2)])
                        if jobs:
                            jobs.pop(0)()
                        sched_barrier()
                    for jb in jobs:
                        jb()
                    u_next = _u_f32(win[0] if stages > 2 else new_raw)
                    if const_expr(chain):
                        ex_next = [EX[kt] if skeleton else EX[kt] * d4s[kt] for kt in range_constexpr(8)]
                    else:
                        ex_next = [Vec(ob[vt][nt_].to(fx.BFloat16).bitcast(fx.Int32)) for vt in range_constexpr(nvt) for nt_ in range_constexpr(4)]
                    if const_expr(ckpt):
                        hit = ((ci + lo + 1) * BT == ck_off) & (ck_row >= 0) & (ck_off > 0)
                        ck_r = _fp(arg_ck + fx.Int64(fx.max(ck_row, fx.Int32(0))) * i64_ck_row_bytes + fx.Int64(h) * STATE_BYTES, STATE_BYTES)
                        for vt in range_constexpr(nvt):
                            for kt in range_constexpr(8):
                                buffer_ops.buffer_store(s_new[vt][kt], ck_r, (16 * (vs0 + vt) + l16) * D + 16 * kt + 4 * qg, mask=hit)
                    results = yield ([fx.as_ir_value(s_new[vt][kt]) for vt in range_constexpr(nvt) for kt in range_constexpr(8)]
                                     + [fx.as_ir_value(x) for x in u_next] + [fx.as_ir_value(x) for x in ex_next]
                                     + [fx.as_ir_value(y) for w in (win[1:] + [new_raw] if stages > 2 else []) for y in w])
                fx.rocdl.s_waitcnt(vmcnt=0)
                fin = [[Vec(results[vt * 8 + kt]) for kt in range_constexpr(8)] for vt in range_constexpr(nvt)]
                if const_expr(chain):
                    aprod = [Vec(results[nvt * 12 + kt]) for kt in range_constexpr(8)]
                    gb_r = _fp(arg_gb + fx.Int64(nhg) * STATE_BYTES, STATE_BYTES)
                    ga_r = _fp(arg_ga + fx.Int64(nhg) * (D * 4), D * 4)
                    gc_r = _fp(arg_gc + fx.Int64(nhg) * FRAGS_C, FRAGS_C)
                    if real:
                        for vt in range_constexpr(nvt):
                            for kt in range_constexpr(8):
                                buffer_ops.buffer_store(fin[vt][kt], gb_r, (16 * (vs0 + vt) + l16) * D + 16 * kt + 4 * qg)
                    else:
                        # C = M - diag(A) as bf16 into LDS, C^T [k'][j]; then the fold's A fragments
                        for vt in range_constexpr(nvt):
                            for kt in range_constexpr(8):
                                v4 = fin[vt][kt]
                                for e in range_constexpr(4):
                                    val = v4[e]
                                    diag = (xs + vt == kt) & (4 * qg + e == l16)
                                    val = diag.select(val - aprod[kt][e], val)
                                    jl = 16 * (wave * nvt + vt) + l16
                                    kk = 16 * kt + 4 * qg + e
                                    word = Vec.from_elements([val, val], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
                                    lds_store_i16(lds_base + (kk * ct_pitch + jl) * 2, fx.Int32(word[0]))
                        fx.rocdl.s_waitcnt(lgkmcnt=0)
                        barrier()
                        ksl = nsl // 2
                        for i in range_constexpr(-(-(8 * ksl) // waves)):
                            f = wave + i * waves
                            if f < 8 * ksl:
                                ktp = f // ksl
                                kl = f % ksl
                                kk = 16 * ktp + l16
                                lo4 = lds_load2(lds_base + (kk * ct_pitch + 32 * kl + 4 * qg) * 2)
                                hi4 = lds_load2(lds_base + (kk * ct_pitch + 32 * kl + 16 + 4 * qg) * 2)
                                frag = Vec.from_elements([lo4[0], lo4[1], hi4[0], hi4[1]], fx.Int32)
                                ksg = (sg * nsl - 8) // 2 + kl
                                buffer_ops.buffer_store(frag, gc_r, ((ktp * 4 + ksg) * 64 + lane) * 4)
                        if (vs0 == 8) & (l16 == 0):
                            for kt in range_constexpr(8):
                                buffer_ops.buffer_store(aprod[kt], ga_r, 16 * kt + 4 * qg)
                else:
                    last = g == gn - 1
                    if last:
                        dst_r = _fp(arg_state + fx.Int64(slot) * i64_slot_bytes + fx.Int64(h) * STATE_BYTES, STATE_BYTES)
                        for vt in range_constexpr(nvt):
                            for kt in range_constexpr(8):
                                buffer_ops.buffer_store(fin[vt][kt], dst_r, (16 * (vs0 + vt) + l16) * D + 16 * kt + 4 * qg)
                    lrow = ((nch - 1) * BT * (i32_ostride * 2)) >> 2
                    for vt in range_constexpr(nvt):
                        for nt_ in range_constexpr(4):
                            buffer_ops.buffer_store(Vec(results[nvt * 12 + vt * 4 + nt_]), out_r, out_lane + lrow + ((nt_ * 16 * (i32_ostride * 2)) >> 2) + 8 * vt)

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
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        arg_gb: fx.Int64,
        arg_gc: fx.Int64,
        arg_ga: fx.Int64,
        arg_sin: fx.Int64,
        i32_groups: fx.Int32,
        i32_minc: fx.Int32,
        i32_npairs: fx.Int32,
        arg_ck: fx.Int64,
        i64_ck_row_bytes: fx.Int64,
        arg_ckoff: fx.Int64,
        arg_ckrow: fx.Int64,
        i32_nbfold: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_out, i32_ostride, arg_gb, arg_gc,
               arg_ga, arg_sin, i32_groups, i32_minc, i32_npairs, arg_ck, i64_ck_row_bytes, arg_ckoff, arg_ckrow, i32_nbfold).launch(
            grid=(i32_nbfold + ((i32_npairs * i32_groups + 7) // 8) * 8 * ngrp, 1, 1), block=(nthreads, 1, 1), stream=stream)

    return launch
