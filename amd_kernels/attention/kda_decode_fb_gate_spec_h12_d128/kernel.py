# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA spec-verify decode for the (state_len, dim) conv state layout: one workgroup per (sequence, head), the delta rule
over the sequence's tokens with the 128 x 128 fp32 state in registers, per-token state snapshots, gated RMSNorm.

Memory instructions are the prologue's limiter (a CU's texture pipe takes ~16 cycles per scattered wave load), so every
small input is loaded as 16-byte lines by all lanes, once per workgroup, into LDS: the q / k / v / gate rows of the
sequence's tokens, f_a, the three conv history rows and the conv weights; the new conv rows are copied back from LDS as
16-byte lines. Stage 1 runs the gate GEMM on the matrix cores (f_a from LDS, f_b from global) and the conv + SiLU from
LDS; stage 2 normalizes q / k and builds the output scale per token (one wave per token); the token loop then has no
barrier: lane (g, s) of wave w owns state rows 8 w' + g (rpl per lane) and columns 4 (s + lpr m) .. + 3, the row
reductions are DPP butterflies, each wave streams its own snapshots (their store issue is the loop's pace), and the
output norm (sum of squares across waves) is one barrier after the last token.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import (
    LOG2E,
    bf16x2_hi,
    bf16x2_lo,
    exp2,
    fma,
    mfma_bf16,
    pk_bf16,
    round_bf16,
    rsqrt,
    sigmoid,
    sum_group,
    wave_sum_lane63,
)
NH = 12
D = 128
LP = NH * D
COL_GATE = 3 * LP
COL_FA = 4 * LP
COL_BETA = COL_FA + D
SCALE = D ** -0.5
EPS_QK = 1e-6


def bf16_i16(v):
    """fp32 of a bf16 read as an i16."""
    return ((fx.Int32(v) & 0xFFFF) << 16).bitcast(fx.Float32)


def compile_kda_decode(max_tok, nw_f32=False, waves=8, lpr=8, store_cm=2, snap=False, uniform=False):
    block = 64 * waves
    rg = 64 // lpr
    rpl = D // (waves * rg)
    cpl = D // lpr
    nvec = cpl // 4
    nt_n = D // 16 // waves
    tk = max_tok
    n_p = tk * 5 * 16  # 16-byte lines of q, k, v, og, f_a per token
    n_h = 3 * 3 * 16  # conv history rows: 3 sets x 3 positions
    n_c = 3 * 4 * 32  # conv weights: 3 sets x 4 taps x 128 / 4
    n_kx = 3 * tk * 16  # new conv rows from the inputs
    n_kh = 3 * 2 * 16  # new conv rows from the history
    r_p = -(-n_p // block)
    r_c = -(-n_c // block)
    r_kx = -(-n_kx // block)
    name = f"kda_decode_fb_gate_spec_h12_d128_t{max_tok}_w{waves}_l{lpr}_cm{store_cm}" + ("_snap" if snap else "") + (
        "_uni" if uniform else "") + ("_nwf32" if nw_f32 else "")

    @fx.struct
    class Shared:
        px: fx.Array[fx.Int32, (tk + 3) * 5 * 64]  # [row][q, k, v, og, f_a][128] bf16: rows 0-2 conv history, row 3 + j token j
        cw: fx.Array[fx.Float32, 12 * D]  # [set][tap][128]
        sets: fx.Array[fx.Float32, tk * 5 * D]  # [token][q, k, v, a, scale][128]
        scal: fx.Array[fx.Float32, 2 * tk]
        part: fx.Array[fx.Float32, tk * waves]
        obuf: fx.Array[fx.Float32, tk * D]

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def kernel(
        arg_proj: fx.Int64,
        i32_pstride: fx.Int32,
        i32_proj_bytes: fx.Int32,
        arg_fbw: fx.Int64,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i64_cs_slot_bytes: fx.Int64,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_state: fx.Int64,
        i64_state_slot_bytes: fx.Int64,
        arg_cu: fx.Int64,
        arg_idx: fx.Int64,
        i32_idx_stride: fx.Int32,
        arg_acc: fx.Int64,
        arg_snap: fx.Int64,
        arg_nw: fx.Int64,
        arg_out: fx.Int64,
        i32_out_stride: fx.Int32,
        i32_out_bytes: fx.Int32,
        f32_lbc: fx.Float32,
        f32_eps: fx.Float32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds = smem.sets.ptr
        scal = smem.scal.ptr
        part = smem.part.ptr
        obuf = smem.obuf.ptr
        atom_f = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        atom_i = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)

        def view(ptr, n):
            return fx.logical_divide(fx.make_view(ptr, fx.make_layout(n, 1)), fx.make_layout(4, 1))

        sets_v = view(smem.sets.ptr, tk * 5 * D)
        px_v = view(smem.px.ptr, (tk + 3) * 5 * 64)
        cw_v = view(smem.cw.ptr, 12 * D)

        def ld4f(v, idx):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
            fx.copy(atom_f, fx.slice(v, (None, idx)), r)
            return Vec(r.load())

        def ld4i(v, idx):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy(atom_i, fx.slice(v, (None, idx)), r)
            return Vec(r.load())

        def st4(v, idx, vec, atom, dt):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), dt)
            r.store(vec)
            fx.copy(atom, r, fx.slice(v, (None, idx)))

        def ld4(idx):
            return ld4f(sets_v, idx)

        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        i_n = bid // NH
        hv = bid % NH
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        s = lane % lpr
        rgi = lane // lpr

        def loadn(rsrc, off, n, dtype):
            x = buffer_ops.buffer_load(rsrc, off, vec_width=n, dtype=dtype)
            out = [x]
            if const_expr(n > 1):
                xv = Vec(x)
                out = [xv[k] for k in range_constexpr(n)]
            return out

        def put(ptr, idx, val):
            ptr[idx] = val

        def readlane63(x):
            return fx.Float32(fx.Int32(fx.rocdl.readlane(
                T.i32, fx.as_ir_value(x.bitcast(fx.Int32)), fx.as_ir_value(fx.Int32(63)))).bitcast(fx.Float32))

        def sload(rsrc, off):
            return fx.Int32(buffer_ops.buffer_load(rsrc, off, vec_width=1, is_scalar=True))

        cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
        idx_r = buffer_ops.create_buffer_resource_from_addr(arg_idx)
        acc_r = buffer_ops.create_buffer_resource_from_addr(arg_acc)
        if const_expr(uniform):
            n0 = i_n * tk
            n_tok = fx.Int32(tk)
        else:
            n0 = sload(cu_r, i_n)
            n_tok = sload(cu_r, i_n + 1) - n0
        n_acc = sload(acc_r, i_n)
        snap_pos = sload(buffer_ops.create_buffer_resource_from_addr(arg_snap), i_n) if snap else fx.Int32(0)
        slots = []
        for h in range_constexpr((tk + 3) // 4):
            sv = Vec(buffer_ops.buffer_load(idx_r, i_n * i32_idx_stride + 4 * h, vec_width=4, is_scalar=True))
            slots += [fx.Int32(sv[k]) for k in range_constexpr(4)]

        def pick(vals, i):
            r = vals[len(vals) - 1]
            for k in range_constexpr(len(vals) - 2, -1, -1):
                r = (i == k).select(vals[k], r)
            return r

        out_r = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=fx.Int64(i32_out_bytes))
        pr = buffer_ops.create_buffer_resource_from_addr(arg_proj, num_records_bytes=i32_proj_bytes)

        # ---- loads that need only the sequence's first row (16-byte lines, all lanes)
        p_loads = []
        for r_i in range_constexpr(r_p):
            item = tx + r_i * block
            jj = item // 80
            rem = item % 80
            seg = rem // 16
            c16 = rem % 16
            col = seg * LP + (seg < 4).select(hv * D, fx.Int32(0))
            off = ((n0 + jj) * i32_pstride + col) // 2 + c16 * 4
            p_loads.append((item, jj, seg, c16, Vec(buffer_ops.buffer_load(
                pr, off, vec_width=4, dtype=fx.Int32, mask=item < n_p))))
        cwr = buffer_ops.create_buffer_resource_from_addr(arg_cw, num_records_bytes=12 * LP * 4)
        c_loads = []
        for r_i in range_constexpr(r_c):
            item = tx + r_i * block
            si_c = item // 128
            tap = (item // 32) % 4
            c4 = item % 32
            off = (si_c * 4 + tap) * LP + hv * D + c4 * 4
            c_loads.append((item, Vec(buffer_ops.buffer_load(cwr, off, vec_width=4, dtype=fx.Float32, mask=item < n_c))))
        a_exp = fx.math.exp(fx.Float32(buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(arg_alog, num_records_bytes=NH * 4), hv, vec_width=1)))
        fbr = buffer_ops.create_buffer_resource_from_addr(arg_fbw, num_records_bytes=LP * D * 2)
        l16 = lane % 16
        qd = lane // 16
        nts = [w * nt_n + j for j in range_constexpr(nt_n)]
        bfr = [
            [Vec(buffer_ops.buffer_load(
                fbr, ((hv * D + 16 * nt + l16) * D + 32 * kb + 8 * qd) // 2, vec_width=4, dtype=fx.Int32))
                for kb in range_constexpr(4)]
            for nt in nts
        ]
        dtbr = buffer_ops.create_buffer_resource_from_addr(arg_dtb, num_records_bytes=LP * 4)
        dtbs = [fx.Float32(buffer_ops.buffer_load(dtbr, hv * D + 16 * nt + l16, vec_width=1)) for nt in nts]
        bl = buffer_ops.buffer_load(pr, (n0 + tx) * i32_pstride + COL_BETA + hv, vec_width=1, dtype=fx.Int16,
                                    mask=tx < tk)
        nwr = buffer_ops.create_buffer_resource_from_addr(arg_nw, num_records_bytes=D * 4)
        if const_expr(nw_f32):
            nw2 = [fx.Float32(x) for x in loadn(nwr, 2 * lane, 2, fx.Float32)]
        else:
            nwp = loadn(nwr, lane, 1, fx.Int32)[0]
            nw2 = [bf16x2_lo(nwp), bf16x2_hi(nwp)]
        v0 = w * (rg * rpl) + rgi * rpl
        fx.rocdl.sched_barrier(0)

        seed = (n_acc > 1).select(n_acc - 1, fx.Int32(0))
        slot0 = pick(slots, seed)
        cslot = slots[0]
        live = (n_tok > 0).select(slot0, fx.Int32(0)) > 0

        if live:
            hv64 = fx.Int64(hv)
            csr = buffer_ops.create_buffer_resource_from_addr(
                arg_cs + fx.Int64(cslot) * i64_cs_slot_bytes, num_records_bytes=fx.Int64(i64_cs_slot_bytes))
            # history rows seed .. seed + 2 of the three sets
            si_h = tx // 48
            p_h = (tx % 48) // 16
            c16_h = tx % 16
            h_load = Vec(buffer_ops.buffer_load(
                csr, ((seed + p_h) * i32_cs_pos + si_h * LP + hv * D) // 2 + c16_h * 4, vec_width=4, dtype=fx.Int32,
                mask=tx < n_h))
            srcr = buffer_ops.create_buffer_resource_from_addr(
                arg_state + fx.Int64(slot0) * i64_state_slot_bytes + hv64 * (D * D * 4), num_records_bytes=D * D * 4)
            soff = [[(v0 + i) * D + 4 * (s + lpr * m) for m in range_constexpr(nvec)] for i in range_constexpr(rpl)]
            st0 = []
            for i in range_constexpr(rpl):
                for m in range_constexpr(nvec):
                    ld = Vec(buffer_ops.buffer_load(srcr, soff[i][m], vec_width=4, dtype=fx.Float32))
                    st0 += [ld[j] for j in range_constexpr(4)]
            fx.rocdl.sched_barrier(0)

            # ---- to LDS
            for item, jj, seg, c16, vec in p_loads:
                if item < n_p:
                    st4(px_v, ((jj + 3) * 5 + seg) * 16 + c16, vec, atom_i, fx.Int32)
            for item, vec in c_loads:
                if item < n_c:
                    st4(cw_v, item, vec, atom_f, fx.Float32)
            if tx < n_h:
                st4(px_v, (p_h * 5 + si_h) * 16 + c16_h, h_load, atom_i, fx.Int32)
            if tx < tk:
                put(scal, 2 * tx + 1, sigmoid(bf16_i16(bl)))
            fx.gpu.barrier()

            # ---- new conv rows: positions 0, 1 from the history rows 1, 2; position 2 + j from token j's input
            def write_conv_rows():
                if tx < n_kh:
                    si_k = tx // 32
                    r_k = (tx % 32) // 16
                    c16_k = tx % 16
                    v4 = ld4i(px_v, ((1 + r_k) * 5 + si_k) * 16 + c16_k)
                    buffer_ops.buffer_store(v4, csr, (r_k * i32_cs_pos + si_k * LP + hv * D) // 2 + c16_k * 4)
                for ri in range_constexpr(r_kx):
                    item = tx + ri * block
                    si_x = item // (tk * 16)
                    j_x = (item % (tk * 16)) // 16
                    c16_x = item % 16
                    if item < n_kx:
                        v4 = ld4i(px_v, ((j_x + 3) * 5 + si_x) * 16 + c16_x)
                        if j_x < n_tok:
                            buffer_ops.buffer_store(
                                v4, csr, (((2 + j_x) * i32_cs_pos + si_x * LP + hv * D) // 2) + c16_x * 4)

            write_conv_rows()

            # ---- stage 1: gate on the matrix cores (f_a from LDS), conv + SiLU from LDS
            zero4 = Vec.filled(4, 0.0, fx.Float32)
            arow = 2 * (l16 // 4) + l16 % 2
            arow = ((l16 % 4 < 2) & (arow < tk)).select(arow, fx.Int32(0))
            afr = [ld4i(px_v, ((arow + 3) * 5 + 4) * 16 + 4 * kb + qd)
                   for kb in range_constexpr(4)]
            for j in range_constexpr(nt_n):
                acc4 = zero4
                for kb in range_constexpr(4):
                    acc4 = mfma_bf16(afr[kb], bfr[j][kb], acc4)
                for e in range_constexpr(2):
                    tq = 2 * qd + e
                    gate = exp2(f32_lbc * sigmoid(a_exp * (round_bf16(acc4[e]) + dtbs[j])))
                    if tq < tk:
                        put(lds, (tq * 5 + 3) * D + 16 * nts[j] + l16, gate)
            # conv + SiLU of this wave's token (all three sets, channels lane and lane + 64), then its q / k norms,
            # the output scale and the q . k dot: one wave per token, no further barrier
            n_s2 = -(-tk // waves)
            for tj in range_constexpr(n_s2):
                tw = w + tj * waves
                if tw < tk:
                    ys = []
                    for si in range_constexpr(3):
                        wt = [[smem.cw.ptr[(si * 4 + tp) * D + 2 * lane + e] for e in range_constexpr(2)]
                              for tp in range_constexpr(4)]
                        acc = [None, None]
                        for m in range_constexpr(4):
                            wv = smem.px.ptr[((tw + m) * 5 + si) * 64 + lane]
                            xs = [bf16x2_lo(wv), bf16x2_hi(wv)]
                            for e in range_constexpr(2):
                                acc[e] = xs[e] * wt[m][e] if m == 0 else fma(xs[e], wt[m][e], acc[e])
                        ys += [acc[e] * sigmoid(acc[e]) for e in range_constexpr(2)]
                    yq, yk, yv = ys[0:2], ys[2:4], ys[4:6]
                    qq = readlane63(wave_sum_lane63(yq[0] * yq[0] + yq[1] * yq[1]))
                    kk = readlane63(wave_sum_lane63(yk[0] * yk[0] + yk[1] * yk[1]))
                    qk = readlane63(wave_sum_lane63(yq[0] * yk[0] + yq[1] * yk[1]))
                    rq = rsqrt(qq + EPS_QK) * SCALE
                    rk = rsqrt(kk + EPS_QK)
                    base = tw * 5 * D
                    for e in range_constexpr(2):
                        put(lds, base + 2 * lane + e, yq[e] * rq)
                        put(lds, base + D + 2 * lane + e, yk[e] * rk)
                        put(lds, base + 2 * D + 2 * lane + e, yv[e])
                    ogp = smem.px.ptr[((tw + 3) * 5 + 3) * 64 + lane]
                    put(lds, base + 4 * D + 2 * lane, nw2[0] * sigmoid(bf16x2_lo(ogp)))
                    put(lds, base + 4 * D + 2 * lane + 1, nw2[1] * sigmoid(bf16x2_hi(ogp)))
                    if lane == 0:
                        put(scal, 2 * tw, qk * (rq * rk))
            fx.gpu.barrier()

            def tok_vec(tt, si, off, n):
                base4 = (tt * 5 + si) * (D // 4) + off // 4
                out = []
                if const_expr(n >= 4):
                    for h in range_constexpr(n // 4):
                        v4 = ld4(base4 + h)
                        out += [v4[j] for j in range_constexpr(4)]
                else:
                    for i in range_constexpr(n):
                        out.append(lds[(tt * 5 + si) * D + off + i])
                return out

            S = [[Vec.from_elements(st0[(i * nvec + m) * 4:(i * nvec + m) * 4 + 4], fx.Float32)
                  for m in range_constexpr(nvec)] for i in range_constexpr(rpl)]
            ss_all = []
            for t in range_constexpr(tk):
                b4 = t * (5 * D // 4)
                vv = tok_vec(t, 2, v0, rpl)
                qkn = scal[2 * t]
                bt = scal[2 * t + 1]
                # pass 1 over the column chunks: dd = a S and the row dots with k and q (S is dead after this pass)
                dd = [[None] * nvec for _ in range_constexpr(rpl)]
                sk4 = [None] * rpl
                sq4 = [None] * rpl
                for m in range_constexpr(nvec):
                    am = ld4(b4 + 96 + s + lpr * m)
                    km = ld4(b4 + 32 + s + lpr * m)
                    qm = ld4(b4 + s + lpr * m)
                    for i in range_constexpr(rpl):
                        dd[i][m] = S[i][m] * am
                        sk4[i] = dd[i][m] * km if m == 0 else fma(dd[i][m], km, sk4[i])
                        sq4[i] = dd[i][m] * qm if m == 0 else fma(dd[i][m], qm, sq4[i])
                new = [[None] * nvec for _ in range_constexpr(rpl)]
                os = []
                ss = fx.Float32(0.0)
                errs = []
                for i in range_constexpr(rpl):
                    sk = sum_group((sk4[i][0] + sk4[i][1]) + (sk4[i][2] + sk4[i][3]), lpr)
                    sq = sum_group((sq4[i][0] + sq4[i][1]) + (sq4[i][2] + sq4[i][3]), lpr)
                    err = (vv[i] - sk) * bt
                    o = sq + err * qkn
                    errs.append(Vec.from_elements([err] * 4, fx.Float32))
                    os.append(o)
                    ss = ss + o * o
                for m in range_constexpr(nvec):
                    km = ld4(b4 + 32 + s + lpr * m)
                    for i in range_constexpr(rpl):
                        new[i][m] = fma(errs[i], km, dd[i][m])
                slot_t = slots[t]
                keep = (slot_t > 0) & (t < n_tok)
                if const_expr(snap):
                    keep = keep & (t == snap_pos)
                dstr = buffer_ops.create_buffer_resource_from_addr(
                    arg_state + fx.Int64(slot_t) * i64_state_slot_bytes + hv64 * (D * D * 4),
                    num_records_bytes=keep.select(fx.Int32(D * D * 4), fx.Int32(0)))
                for i in range_constexpr(rpl):
                    for m in range_constexpr(nvec):
                        buffer_ops.buffer_store(new[i][m], dstr, soff[i][m], cache_modifier=store_cm)
                S = new
                for i in range_constexpr(rpl):
                    if s == 0:
                        put(obuf, t * D + v0 + i, os[i])
                ss_all.append(ss)

            for t in range_constexpr(tk):
                tot = fx.Int32(fx.rocdl.readlane(
                    T.i32, fx.as_ir_value(wave_sum_lane63(ss_all[t] * fx.Float32(1.0 / lpr)).bitcast(fx.Int32)),
                    fx.as_ir_value(fx.Int32(63)))).bitcast(fx.Float32)
                if lane == 0:
                    put(part, t * waves + w, tot)
            fx.gpu.barrier()
            for t in range_constexpr(tk):
                tsum = part[t * waves]
                for k in range_constexpr(1, waves):
                    tsum = tsum + part[t * waves + k]
                rstd = rsqrt(tsum * fx.Float32(1.0 / D) + f32_eps)
                scn = tok_vec(t, 4, v0, rpl)
                ow = [obuf[t * D + v0 + i] * rstd * scn[i] for i in range_constexpr(rpl)]
                packed = [pk_bf16(ow[2 * k], ow[2 * k + 1]) for k in range_constexpr(rpl // 2)]
                buffer_ops.buffer_store(
                    Vec.from_elements(packed, fx.Int32) if rpl > 2 else packed[0], out_r,
                    ((n0 + t) * i32_out_stride + hv * D + v0) // 2, mask=(s == 0) & (t < n_tok))
        else:
            for t in range(fx.Index(0), fx.Index(n_tok), fx.Index(1)):
                if tx < 16:
                    buffer_ops.buffer_store(
                        Vec.from_elements([fx.Int32(0)] * 4, fx.Int32), out_r,
                        ((n0 + fx.Int32(t)) * i32_out_stride + hv * D) // 2 + 4 * tx)

    @flyc.jit
    def launch(
        arg_proj: fx.Int64,
        i32_pstride: fx.Int32,
        i32_proj_bytes: fx.Int32,
        arg_fbw: fx.Int64,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i64_cs_slot_bytes: fx.Int64,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_state: fx.Int64,
        i64_state_slot_bytes: fx.Int64,
        arg_cu: fx.Int64,
        arg_idx: fx.Int64,
        i32_idx_stride: fx.Int32,
        arg_acc: fx.Int64,
        arg_snap: fx.Int64,
        arg_nw: fx.Int64,
        arg_out: fx.Int64,
        i32_out_stride: fx.Int32,
        i32_out_bytes: fx.Int32,
        f32_lbc: fx.Float32,
        f32_eps: fx.Float32,
        i32_nseq: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_proj, i32_pstride, i32_proj_bytes, arg_fbw, arg_alog, arg_dtb, arg_cw, arg_cs, i64_cs_slot_bytes,
            i32_cs_dim, i32_cs_pos, arg_state, i64_state_slot_bytes, arg_cu, arg_idx, i32_idx_stride, arg_acc,
            arg_snap, arg_nw, arg_out, i32_out_stride, i32_out_bytes, f32_lbc, f32_eps,
        ).launch(grid=(fx.Int64(i32_nseq) * NH, 1, 1), block=(block, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
