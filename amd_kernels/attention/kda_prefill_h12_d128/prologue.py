# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA prefill prologue: conv(4) + silu, q / k L2 norm, gate cumsum, the intra-chunk matrices and the workspace records.

One workgroup (256 threads) per (chunk, head). A thread owns 8 channels of 4 consecutive tokens: thread
``(cg, run)`` with ``run = 8 ks + 4 hf + q`` holds tokens ``4 run .. 4 run + 3``, so its tokens are one 4-run of the
fragment-major workspace layout (reference.py) and the exchange that completes a 16-byte fragment is one lane swap.

Phases (g stays in registers; LDS holds only what the matrix cores need):
  1. gate: raw g -> glog -> chunk cumsum in log2 units (per-thread scan, prefix over the 16 runs through LDS).
  2. k band: conv + silu, L2 norm; Kd / Ki (decay relative to the first row of the token's 16-row block, so exp2 stays
     in range), kg (stored fragment-major straight from registers), kb^T (bf16 operand of the w product).
  3. Akk = Kd Ki^T per 16x16 block pair (off-diagonal blocks scaled by the bridge exp2(gn_i - gn_j) per channel).
  4. q band: Qd into the Kd tile, qg stored; Aqk likewise, stored from the MFMA accumulators.
  5. (I + L)^-1: fp32 forward substitution on the 16x16 diagonal blocks, off-diagonal blocks merged by fp32 MFMAs.
  6. w^T = kb^T A^T and u = A (beta v) on the matrix cores, stored fragment-major.
Padded rows (past the sequence end) are zero in every tile. The chunk-0 workgroup of a sequence reads the conv history
and rolls conv_state.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.rocdl import _split_mfma_operands
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import launcher

from .common import FLAG_STRIDE, WALK_GROUPS

NH = 12
D = 128
LP = NH * D
BT = 64
LOG2E = 1.4426950408889634
RCP_LN2 = 1.4426950408889634
EPS_QK = 1e-6
REC_W, REC_QG, REC_KGT, REC_AQK, REC_DEC, REC_U, RECORD_BYTES = 0, 16384, 32768, 49152, 57344, 58368, 74752

PITCH_T = 272  # [64][136] bf16 rows (Kd, Ki)
PITCH_C = 136  # [128][68] bf16 rows (kb^T, vb^T)
SZ_T = BT * PITCH_T
R_X = 0  # Kd, then Qd, then inverse blocks and scratch, then kb^T
R_Y = R_X + SZ_T  # Ki, then A (bf16 blocks), vb^T from R_Y + 8192
R_L = R_Y + SZ_T  # L blocks (fp32)
R_GN = R_L + 10 * 1024
R_GL = R_GN + 4 * D * 4
R_BETA = R_GL + D * 4
R_BR = R_BETA + BT * 4
R_HIST = R_BR + 6 * D * 4
R_TOTAL = R_HIST + 9 * D * 2
R_RT = R_X  # run totals of the scan, dead before Kd is written
R_AINV = R_X
R_SCR = R_AINV + 10 * 1024
R_A = R_Y
R_KBT = R_X
R_VBT = R_Y + 8192
R_KD, R_KI = R_X, R_Y
LDS_BYTES = R_TOTAL

PAIRS = [(i, j) for i in range(4) for j in range(i + 1)]
OFFD = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]


def raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def lds_ptr(addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), raw(fx.Int64(addr)))


def lds_ld(addr, ty, align):
    return llvm.load(ty, lds_ptr(addr), alignment=align)


def lds_st(val, addr, align):
    llvm.store(raw(val), lds_ptr(addr), alignment=align)




def ld4f(addr):
    return Vec(lds_ld(addr, T.vec(4, T.f32), 16))


def ld4i(addr):
    return Vec(lds_ld(addr, T.vec(4, T.i32), 16))


def ld2i(addr):
    return Vec(lds_ld(addr, T.vec(2, T.i32), 8))


def ldf(addr):
    return fx.Float32(lds_ld(addr, T.f32, 4))


def st4(vals, addr, dt):
    lds_st(Vec.from_elements(list(vals), dt), addr, 16)


def st2i(vals, addr):
    lds_st(Vec.from_elements(list(vals), fx.Int32), addr, 8)


def stf(val, addr):
    lds_st(fx.Float32(val), addr, 4)


def barrier():
    fx.gpu.barrier()
    fx.rocdl.sched_barrier(0)


def pack_bf16(vals):
    """fp32 values -> bf16 (RNE) packed two per dword; a compiler-visible conversion, so MFMA hazards are handled."""
    n = len(vals)
    v = Vec.from_elements([fx.Float32(x) for x in vals], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
    return [fx.Int32(v[k]) for k in range(n // 2)]


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (RNE) by inline asm (cheaper to schedule than the compiler's conversion)."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def bf_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf_hi(w):
    return (fx.Int32(w) & -65536).bitcast(fx.Float32)


def round_bf16(x):
    bits = fx.Float32(x).bitcast(fx.Int32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & -65536).bitcast(fx.Float32)


def exp2(x):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.as_ir_value(fx.Float32(x))))


def rcp(x):
    return fx.Float32(fx.rocdl.rcp(T.f32, fx.as_ir_value(fx.Float32(x))))


def rsqrt(x):
    return fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(fx.Float32(x))))


def sigmoid(x):
    return rcp(fx.Float32(1.0) + exp2(x * fx.Float32(-LOG2E)))


def fma(a, b, c):
    return fx.math.fma(a, b, c)


def dpp(v, ctrl, row_mask=0xF, old=None):
    from flydsl._mlir.dialects import rocdl as _rocdl

    bits = fx.Float32(v).bitcast(fx.Int32)
    o = llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(fx.Float32(old).bitcast(fx.Int32))
    r = _rocdl.update_dpp(T.i32, o, fx.as_ir_value(bits), ctrl, row_mask, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def dpp_i(v, ctrl):
    from flydsl._mlir.dialects import rocdl as _rocdl

    r = _rocdl.update_dpp(T.i32, llvm.mlir_undef(T.i32), fx.as_ir_value(fx.Int32(v)), ctrl, 0xF, 0xF, True)
    return fx.Int32(r)


def sum16(v):
    for ctrl in (0xB1, 0x4E, 0x141, 0x140):
        v = v + dpp(v, ctrl)
    return v


def dot2(a, b, c):
    """c + a.lo * b.lo + a.hi * b.hi for bf16 pairs in dwords (fp32 accumulate)."""
    av = Vec.from_elements([fx.Int32(a)], fx.Int32).bitcast(fx.BFloat16)
    bv = Vec.from_elements([fx.Int32(b)], fx.Int32).bitcast(fx.BFloat16)
    return fx.Float32(fx.rocdl.fdot2_f32_bf16_(T.f32, raw(av), raw(bv), fx.as_ir_value(fx.Float32(c))))


def mfma_bf16(a4, b4, acc):
    a = Vec(a4).bitcast(fx.BFloat16)
    b = Vec(b4).bitcast(fx.BFloat16)
    return Vec(fx.rocdl.mfma_f32_16x16x32_bf16(Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]))


def mfma_f32(a, b, c):
    a_, b_, c_, cbsz, abid, blgp = _split_mfma_operands([a, b, c, 0, 0, 0])
    return Vec(fx.rocdl.mfma_f32_16x16x4f32_(Vec.make_type(4, fx.Float32), a_, b_, c_, cbsz, abid, blgp))


def swap32(a, b):
    pair = fx.rocdl.permlane32_swap(
        llvm.StructType.get_literal([T.i32, T.i32]), fx.as_ir_value(fx.Int32(a)), fx.as_ir_value(fx.Int32(b)), False, False
    )
    return fx.Int32(llvm.extractvalue(T.i32, pair, [0])), fx.Int32(llvm.extractvalue(T.i32, pair, [1]))


def unpack8(v4):
    out = []
    for k in range(4):
        out += [bf_lo(v4[k]), bf_hi(v4[k])]
    return out


def pack8(vals):
    return [pk_bf16(vals[2 * k], vals[2 * k + 1]) for k in range(len(vals) // 2)]


@ASTRewriter.transform
def prologue_body(
    base,
    cid,
    h,
    rot,
    arg_mixed,
    i32_ldx,
    i32_mixed_bytes,
    arg_rawg,
    i32_ldg,
    arg_beta,
    i32_ldb,
    arg_alog,
    arg_dtb,
    arg_cw,
    arg_cs,
    i32_cs_slot,
    i32_cs_dim,
    i32_cs_pos,
    arg_tab,
    arg_ws,
    i32_nc,
    f32_lb,
    f32_scale,
    wcm=0,
    arg_flags=None,
):
    """One workgroup's work: the workspace record of chunk ``cid`` and head ``h``. ``wcm``: cache modifier of the record stores;
    ``arg_flags``: when given, the record is published (two flags of the chunk's walk workgroups) after its stores completed."""
    tx = fx.Int32(fx.thread_idx.x)
    lane = tx % 64
    wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
    lg = lane // 16
    cg = lane % 16
    hf = lg // 2
    qq = 2 * (wave % 2) + (lg % 2)
    ks_t = wave // 2
    role = (wave + rot) % 4
    run = 8 * ks_t + 4 * hf + qq
    r0 = 4 * run

    def sload(rsrc, off):
        return fx.Int32(buffer_ops.buffer_load(rsrc, off, vec_width=1, is_scalar=True))

    tab_r = buffer_ops.create_buffer_resource_from_addr(arg_tab, num_records_bytes=fx.Int32(0x7FFFFFF0))
    row = Vec(buffer_ops.buffer_load(tab_r, cid * 4, vec_width=4, is_scalar=True))
    bos, tn, t0, enc = fx.Int32(row[0]), fx.Int32(row[1]), fx.Int32(row[2]), fx.Int32(row[3])
    rows = fx.min(fx.Int32(64), tn - t0)
    slot = enc & 0x3FFFFFFF
    has_init = (enc >> 30) == 0
    first = t0 == 0

    mr = buffer_ops.create_buffer_resource_from_addr(arg_mixed, num_records_bytes=i32_mixed_bytes)
    gr = buffer_ops.create_buffer_resource_from_addr(arg_rawg, num_records_bytes=fx.Int32(0x7FFFFFF0))
    br = buffer_ops.create_buffer_resource_from_addr(arg_beta, num_records_bytes=fx.Int32(0x7FFFFFF0))
    cwr = buffer_ops.create_buffer_resource_from_addr(arg_cw, num_records_bytes=3 * 4 * LP * 4)
    # the sequence's conv-state slot as its own resource, based with a 64-bit address: vLLM's hybrid cache pages give a
    # slot stride of ~10M bf16 (one ~21 MB page), so a 32-bit `slot * stride` offset overflows past slot ~100
    csr = buffer_ops.create_buffer_resource_from_addr(
        arg_cs + fx.Int64(slot) * fx.Int64(i32_cs_slot) * fx.Int64(2), num_records_bytes=fx.Int32(0x7FFFFFF0)
    )
    wsr = buffer_ops.create_buffer_resource_from_addr(
        arg_ws + fx.Int64(h * i32_nc + cid) * fx.Int64(RECORD_BYTES), num_records_bytes=RECORD_BYTES
    )

    # ---- conv history of the sequence's slot (chunk 0 only) -> LDS [band][pos][128]
    hit = tx < 144
    hb_i = tx // 48
    hp_i = (tx % 48) // 16
    hc = tx % 16
    hv = []
    for e in range_constexpr(8):
        ch = hb_i * LP + h * D + hc * 8 + e
        off = ch * i32_cs_dim + hp_i * i32_cs_pos
        v = buffer_ops.buffer_load(csr, off, vec_width=1, dtype=fx.Int16, mask=hit & first & has_init)
        hv.append(fx.Int32(v) & 0xFFFF)
    if hit:
        st4([hv[2 * k] | (hv[2 * k + 1] << 16) for k in range_constexpr(4)], base + R_HIST + tx * 16, fx.Int32)

    def band_rows(band):
        rowsv = []
        for j in range_constexpr(7):
            s = t0 + r0 - 3 + j
            valid = (s >= 0) & (s < tn)
            rowsv.append(
                Vec(buffer_ops.buffer_load(mr, ((bos + s) * i32_ldx + band * LP + h * D + cg * 8) // 2, vec_width=4, dtype=fx.Int32, mask=valid))
            )
        return rowsv

    def band_wts(band, half):
        return [Vec(buffer_ops.buffer_load(cwr, (band * 4 + tap) * LP + h * D + cg * 8 + 4 * half, vec_width=4)) for tap in range_constexpr(4)]

    # ---- loads that start the HBM latency early: k band, gate, beta
    dtr = buffer_ops.create_buffer_resource_from_addr(arg_dtb, num_records_bytes=LP * 4)
    dtb = []
    for k in range_constexpr(2):
        v4 = Vec(buffer_ops.buffer_load(dtr, h * D + cg * 8 + 4 * k, vec_width=4))
        dtb += [v4[c] for c in range_constexpr(4)]
    alr = buffer_ops.create_buffer_resource_from_addr(arg_alog, num_records_bytes=NH * 4)
    a_exp = fx.math.exp(fx.Float32(buffer_ops.buffer_load(alr, h, vec_width=1)))
    graw = []
    for j in range_constexpr(4):
        live = (r0 + j) < rows
        graw.append(Vec(buffer_ops.buffer_load(gr, ((bos + t0 + r0 + j) * i32_ldg + h * D + cg * 8) // 2, vec_width=4, dtype=fx.Int32, mask=live)))
    betas = []
    for j in range_constexpr(4):
        bl = buffer_ops.buffer_load(br, (bos + t0 + r0 + j) * i32_ldb + h, vec_width=1, dtype=fx.Int16, mask=(r0 + j) < rows)
        betas.append(((r0 + j) < rows).select(sigmoid(((fx.Int32(bl) & 0xFFFF) << 16).bitcast(fx.Float32)), fx.Float32(0.0)))
    k_rows = band_rows(1)

    # ---- gate: glog, chunk cumsum (log2 units)
    gl = []
    for j in range_constexpr(4):
        vals = unpack8(graw[j])
        gl.append([f32_lb * sigmoid(a_exp * (vals[c] + dtb[c])) for c in range_constexpr(8)])
    gc = [gl[0]]
    for j in range_constexpr(1, 4):
        gc.append([gc[j - 1][c] + gl[j][c] for c in range_constexpr(8)])
    st4(gc[3][0:4], base + R_RT + (run * D + cg * 8) * 4, fx.Float32)
    st4(gc[3][4:8], base + R_RT + (run * D + cg * 8) * 4 + 16, fx.Float32)
    barrier()
    if tx < D:
        tots = [ldf(base + R_RT + (r * D + tx) * 4) for r in range_constexpr(16)]
        acc_p = fx.Float32(0.0)
        for r in range_constexpr(16):
            stf(acc_p, base + R_RT + (r * D + tx) * 4)
            acc_p = acc_p + tots[r]
    barrier()
    pa = ld4f(base + R_RT + (run * D + cg * 8) * 4)
    pb = ld4f(base + R_RT + (run * D + cg * 8) * 4 + 16)
    pref = [pa[c] for c in range_constexpr(4)] + [pb[c] for c in range_constexpr(4)]
    g = [[(pref[c] + gc[j][c]) * RCP_LN2 for c in range_constexpr(8)] for j in range_constexpr(4)]

    if (run % 4) == 0:
        st4(g[0][0:4], base + R_GN + ((run // 4) * D + cg * 8) * 4, fx.Float32)
        st4(g[0][4:8], base + R_GN + ((run // 4) * D + cg * 8) * 4 + 16, fx.Float32)
    last = rows - 1
    if run == last // 4:
        jl = last % 4
        for c in range_constexpr(8):
            lastv = (jl == 0).select(g[0][c], (jl == 1).select(g[1][c], (jl == 2).select(g[2][c], g[3][c])))
            stf(lastv, base + R_GL + (cg * 8 + c) * 4)
    if cg == 0:
        for j in range_constexpr(4):
            stf(betas[j], base + R_BETA + (r0 + j) * 4)
    barrier()
    glast = [ldf(base + R_GL + (cg * 8 + c) * 4) for c in range_constexpr(8)]
    gnb = ld4f(base + R_GN + ((run // 4) * D + cg * 8) * 4)
    gnb2 = ld4f(base + R_GN + ((run // 4) * D + cg * 8) * 4 + 16)
    gn = [gnb[c] for c in range_constexpr(4)] + [gnb2[c] for c in range_constexpr(4)]

    if tx < 32:
        lv = ld4f(base + R_GL + tx * 16)
        buffer_ops.buffer_store(Vec.from_elements([exp2(lv[c]) for c in range_constexpr(4)], fx.Float32), wsr, REC_DEC // 4 + tx * 4, cache_modifier=wcm)

    # bridge table exp2(gn_i - gn_j) of the six off-diagonal block pairs
    for u in range_constexpr(3):
        p = tx // 128 + 2 * u
        c = tx % 128
        ib = (p == 0).select(fx.Int32(1), (p <= 2).select(fx.Int32(2), fx.Int32(3)))
        jb = (p == 0).select(fx.Int32(0), (p == 1).select(fx.Int32(0), (p == 2).select(fx.Int32(1), (p == 3).select(fx.Int32(0), (p == 4).select(fx.Int32(1), fx.Int32(2))))))
        stf(exp2(ldf(base + R_GN + (ib * D + c) * 4) - ldf(base + R_GN + (jb * D + c) * 4)), base + R_BR + (p * D + c) * 4)

    def band_conv(band, rowsv):
        """conv + silu of the thread's 4 tokens x 8 channels, as bf16 pairs [row][4]."""
        words = []
        for j in range_constexpr(7):
            v4 = [rowsv[j][k] for k in range_constexpr(4)]
            if j < 3:
                s = t0 + r0 - 3 + j
                pos = fx.min(fx.max(s + 3, fx.Int32(0)), fx.Int32(2))
                hv4 = ld4i(base + R_HIST + (band * 3 + pos) * 256 + cg * 16)
                v4 = [(s < 0).select(hv4[k], v4[k]) for k in range_constexpr(4)]
            words.append(v4)
        yp = [[] for _ in range_constexpr(4)]
        for half in range_constexpr(2):
            wts = band_wts(band, half)
            ypre = [[None] * 4 for _ in range_constexpr(4)]
            for cc in range_constexpr(4):
                c = 4 * half + cc
                xc = [bf_hi(words[j][c // 2]) if c % 2 else bf_lo(words[j][c // 2]) for j in range_constexpr(7)]
                for r in range_constexpr(4):
                    acc = xc[r] * wts[0][cc]
                    for tap in range_constexpr(1, 4):
                        acc = fma(xc[r + tap], wts[tap][cc], acc)
                    ypre[r][cc] = acc * rcp(fx.Float32(1.0) + exp2(acc * fx.Float32(-LOG2E)))
            for r in range_constexpr(4):
                yp[r] += pack_bf16(ypre[r])
        return yp

    def row_scale(yp):
        out = []
        for r in range_constexpr(4):
            ss = fx.Float32(0.0)
            for w in yp[r]:
                ss = dot2(w, w, ss)
            out.append(((r0 + r) < rows).select(rsqrt(sum16(ss) + EPS_QK), fx.Float32(0.0)))
        return out

    def unp(yp_row):
        return unpack8(yp_row)

    def tile_store(res, vals_per_row):
        for r in range_constexpr(4):
            st4(pack8(vals_per_row[r]), base + res + (r0 + r) * PITCH_T + cg * 16, fx.Int32)

    # ---- k band
    cbk = [exp2(fx.min(glast[c] - gn[c], fx.Float32(0.0))) for c in range_constexpr(8)]
    egn = [exp2(gn[c]) for c in range_constexpr(8)]
    yp = band_conv(1, k_rows)
    rk = row_scale(yp)
    kg_pairs, kb_pairs = [], []
    for rp in range_constexpr(2):
        kgv, kbv = [], []
        for rr in range_constexpr(2):
            r = 2 * rp + rr
            yv = unp(yp[r])
            kn = [yv[c] * rk[r] for c in range_constexpr(8)]
            e1 = [exp2(g[r][c] - gn[c]) for c in range_constexpr(8)]
            e2 = [exp2(gn[c] - g[r][c]) for c in range_constexpr(8)]
            st4(pack8([kn[c] * e1[c] for c in range_constexpr(8)]), base + R_KD + (r0 + r) * PITCH_T + cg * 16, fx.Int32)
            st4(pack8([kn[c] * e2[c] for c in range_constexpr(8)]), base + R_KI + (r0 + r) * PITCH_T + cg * 16, fx.Int32)
            kgv.append([kn[c] * (e2[c] * cbk[c]) for c in range_constexpr(8)])
            kbv.append([kn[c] * betas[r] * (e1[c] * egn[c]) for c in range_constexpr(8)])
        kg_pairs.append([pk_bf16(kgv[0][c], kgv[1][c]) for c in range_constexpr(8)])
        kb_pairs.append([pk_bf16(kbv[0][c], kbv[1][c]) for c in range_constexpr(8)])
    kb_pk = [[kb_pairs[0][c], kb_pairs[1][c]] for c in range_constexpr(8)]

    # kg straight to the workspace: fragment (mt = cg // 2, ks_t), lane qq * 16 + i, slot half hf
    for pp in range_constexpr(4):
        c = 2 * pp
        a2 = [kg_pairs[0][c], kg_pairs[1][c]]
        b2 = [kg_pairs[0][c + 1], kg_pairs[1][c + 1]]
        sw = [swap32(a2[d], b2[d]) for d in range_constexpr(2)]
        is0 = hf == 0
        recv = [is0.select(sw[d][1], sw[d][0]) for d in range_constexpr(2)]
        own = [is0.select(a2[d], b2[d]) for d in range_constexpr(2)]
        lo = [is0.select(own[d], recv[d]) for d in range_constexpr(2)]
        hi_ = [is0.select(recv[d], own[d]) for d in range_constexpr(2)]
        chan = c + hf
        i_ch = (cg % 2) * 8 + chan
        frag = ((cg // 2) * 2 + ks_t) * 64 + qq * 16 + i_ch
        buffer_ops.buffer_store(Vec.from_elements(lo + hi_, fx.Int32), wsr, (REC_KGT + frag * 16) // 4, cache_modifier=wcm)


    # ---- loads of the q band issued ahead of the Akk matrix phase
    q_rows = band_rows(0)

    barrier()

    def rd_t(res, row, ks):
        a = base + res + row * PITCH_T + (32 * ks + 4 * lg) * 2
        x, y = ld2i(a), ld2i(a + 32)
        return Vec.from_elements([x[0], x[1], y[0], y[1]], fx.Int32)

    def rd_c(res, row, ks):
        a = base + res + row * PITCH_C + (32 * ks + 4 * lg) * 2
        x, y = ld2i(a), ld2i(a + 32)
        return Vec.from_elements([x[0], x[1], y[0], y[1]], fx.Int32)

    def scale_frag(frag, pidx, ks):
        b0 = ld4f(base + R_BR + (pidx * D + 32 * ks + 4 * lg) * 4)
        b1 = ld4f(base + R_BR + (pidx * D + 32 * ks + 4 * lg + 16) * 4)
        vals = unpack8([frag[k] for k in range_constexpr(4)])
        bs = [b0[k] for k in range_constexpr(4)] + [b1[k] for k in range_constexpr(4)]
        return Vec.from_elements(pack_bf16([vals[k] * bs[k] for k in range_constexpr(8)]), fx.Int32)

    def pair_accs(pairs, qd_res):
        """Accumulators of several block pairs, their MFMA chains interleaved."""
        accs = [Vec.filled(4, 0.0, fx.Float32) for _ in pairs]
        for ks in range_constexpr(4):
            for n, (ib, jb) in enumerate(pairs):
                a = rd_t(R_KI, 16 * jb + cg, ks)
                b = rd_t(qd_res, 16 * ib + cg, ks)
                if ib != jb:
                    b = scale_frag(b, OFFD.index((ib, jb)), ks)
                accs[n] = mfma_bf16(a, b, accs[n])
        return accs

    zero = fx.Float32(0.0)
    # Akk -> L (fp32 blocks)
    for w_i in range_constexpr(4):
        if role == w_i:
            mine = list(range(w_i, 10, 4))
            accs_all = pair_accs([PAIRS[pi] for pi in mine], R_KD)
            for pi, acc in zip(mine, accs_all):
                ib, jb = PAIRS[pi]
                bt = ldf(base + R_BETA + (16 * ib + cg) * 4)
                outv = []
                for e in range_constexpr(4):
                    v = acc[e] * bt
                    if ib == jb:
                        v = (4 * lg + e < cg).select(v, zero)
                    outv.append(v)
                st4(outv, base + R_L + pi * 1024 + (cg * 16 + 4 * lg) * 4, fx.Float32)
    barrier()

    # ---- q band and v band
    v_rows = band_rows(2)
    yp = band_conv(0, q_rows)
    rq = row_scale(yp)
    egq = [exp2(gn[c]) for c in range_constexpr(8)]
    hc_ = (cg % 4) // 2
    h0 = hc_ == 0
    for r in range_constexpr(4):
        yv = unp(yp[r])
        qn = [yv[c] * rq[r] * f32_scale for c in range_constexpr(8)]
        e1 = [exp2(g[r][c] - gn[c]) for c in range_constexpr(8)]
        st4(pack8([qn[c] * e1[c] for c in range_constexpr(8)]), base + R_KD + (r0 + r) * PITCH_T + cg * 16, fx.Int32)
        w4 = pack8([qn[c] * (e1[c] * egq[c]) for c in range_constexpr(8)])
        send = [h0.select(w4[2], w4[0]), h0.select(w4[3], w4[1])]
        recv = [dpp_i(send[0], 0x4E), dpp_i(send[1], 0x4E)]
        mine = [h0.select(w4[0], w4[2]), h0.select(w4[1], w4[3])]
        lo = [h0.select(mine[0], recv[0]), h0.select(mine[1], recv[1])]
        hi_ = [h0.select(recv[0], mine[0]), h0.select(recv[1], mine[1])]
        q_c = 2 * (cg % 2) + hc_
        tok16 = 4 * (run % 4) + r
        frag = ((run // 4) * 4 + cg // 4) * 64 + q_c * 16 + tok16
        buffer_ops.buffer_store(Vec.from_elements(lo + hi_, fx.Int32), wsr, (REC_QG + frag * 16) // 4, cache_modifier=wcm)
    yp = band_conv(2, v_rows)
    yv = [unp(yp[r]) for r in range_constexpr(4)]
    vb_pk = []
    for c in range_constexpr(8):
        vb = [yv[r][c] * betas[r] for r in range_constexpr(4)]
        vb_pk.append([pk_bf16(vb[0], vb[1]), pk_bf16(vb[2], vb[3])])
    barrier()

    # Aqk straight to the workspace
    groups = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)]
    for w_i in range_constexpr(4):
        if role == w_i:
            gs = list(range(w_i, 8, 4))
            need = [(groups[gi][0], jb) for gi in gs for jb in range(2 * groups[gi][1], 2 * groups[gi][1] + 2) if jb <= groups[gi][0]]
            accd = dict(zip(need, pair_accs(need, R_KD)))
            for gi in gs:
                ib, ks2 = groups[gi]
                tiles = []
                for jb in range(2 * ks2, 2 * ks2 + 2):
                    if jb > ib:
                        tiles.append([zero] * 4)
                    else:
                        acc = accd[(ib, jb)]
                        vals = [acc[e] for e in range_constexpr(4)]
                        if ib == jb:
                            vals = [(4 * lg + e <= cg).select(vals[e], zero) for e in range_constexpr(4)]
                        tiles.append(vals)
                words = pack_bf16(tiles[0] + tiles[1])
                buffer_ops.buffer_store(Vec.from_elements(words, fx.Int32), wsr, (REC_AQK + ((ib * 2 + ks2) * 64 + lane) * 16) // 4, cache_modifier=wcm)
    barrier()

    # ---- (I + L)^-1: diagonal blocks by forward substitution (wave 0, one block per 16-lane group)
    def lblk(i, j):
        return i * (i + 1) // 2 + j

    if role == 0:
        lb_addr = base + R_L + (lg * (lg + 1) // 2 + lg) * 1024
        xs = []
        for i in range_constexpr(16):
            rowl = []
            for k4 in range_constexpr((i + 3) // 4):
                v4 = ld4f(lb_addr + (i * 16 + 4 * k4) * 4)
                rowl += [v4[e] for e in range_constexpr(4)]
            s0 = fx.Float32(0.0)
            s1 = fx.Float32(0.0)
            for m in range_constexpr(i):
                if m % 2 == 0:
                    s0 = fma(rowl[m], xs[m], s0)
                else:
                    s1 = fma(rowl[m], xs[m], s1)
            xi = (cg == i).select(fx.Float32(1.0), fx.Float32(0.0)) - (s0 + s1)
            xs.append(xi)
        ai_addr = base + R_AINV + (lg * (lg + 1) // 2 + lg) * 1024
        for i in range_constexpr(16):
            stf(xs[i], ai_addr + (i * 16 + cg) * 4)
    barrier()
    def mm_acc(a_addr_fn, b_addr_fn, acc):
        for kk in range_constexpr(0, 16, 4):
            acc = mfma_f32(ldf(a_addr_fn(kk)), ldf(b_addr_fn(kk)), acc)
        return acc

    levels = [[(1, 0), (2, 1), (3, 2)], [(2, 0), (3, 1)], [(3, 0)]]
    for lv in range_constexpr(3):
        for w_i in range_constexpr(len(levels[lv])):
            if role == w_i:
                i_b, j_b = levels[lv][w_i]
                acc = Vec.filled(4, 0.0, fx.Float32)
                for m in range_constexpr(j_b, i_b):
                    acc = mm_acc(
                        lambda kk: base + R_L + lblk(i_b, m) * 1024 + (cg * 16 + kk + lg) * 4,
                        lambda kk: base + R_AINV + lblk(m, j_b) * 1024 + ((kk + lg) * 16 + cg) * 4,
                        acc,
                    )
                scr = base + R_SCR + w_i * 1024
                for e in range_constexpr(4):
                    stf(acc[e], scr + ((4 * lg + e) * 16 + cg) * 4)
                r_ = mm_acc(
                    lambda kk: base + R_AINV + lblk(i_b, i_b) * 1024 + (cg * 16 + kk + lg) * 4,
                    lambda kk: scr + ((kk + lg) * 16 + cg) * 4,
                    Vec.filled(4, 0.0, fx.Float32),
                )
                for e in range_constexpr(4):
                    stf(zero - r_[e], base + R_AINV + lblk(i_b, j_b) * 1024 + ((4 * lg + e) * 16 + cg) * 4)
        barrier()
    # A as bf16 blocks [ib][jb][16][16], zero above the diagonal
    jb_t = tx % 4
    r_t = (tx // 4) % 16
    src_blk = wave * (wave + 1) // 2 + fx.min(jb_t, wave)
    v0 = ld4f(base + R_AINV + src_blk * 1024 + r_t * 64)
    v1 = ld4f(base + R_AINV + src_blk * 1024 + r_t * 64 + 16)
    v2 = ld4f(base + R_AINV + src_blk * 1024 + r_t * 64 + 32)
    v3 = ld4f(base + R_AINV + src_blk * 1024 + r_t * 64 + 48)
    live_b = jb_t <= wave
    fl = [v[e] for v in (v0, v1, v2, v3) for e in range_constexpr(4)]
    fl = [live_b.select(x, zero) for x in fl]
    wd = pack8(fl[0:8]) + pack8(fl[8:16])
    a_addr = base + R_A + (wave * 4 + jb_t) * 512 + r_t * 32
    barrier()
    st4(wd[0:4], a_addr, fx.Int32)
    st4(wd[4:8], a_addr + 16, fx.Int32)
    barrier()
    for c in range_constexpr(8):
        st2i(kb_pk[c], base + R_KBT + (cg * 8 + c) * PITCH_C + r0 * 2)
        st2i(vb_pk[c], base + R_VBT + (cg * 8 + c) * PITCH_C + r0 * 2)
    barrier()

    def rd_a(ib, ks):
        a = base + R_A + ((ib * 4 + 2 * ks) * 16 + cg) * 32 + 8 * lg
        x, y = ld2i(a), ld2i(a + 512)
        return Vec.from_elements([x[0], x[1], y[0], y[1]], fx.Int32)

    # w^T = kb^T A^T: wave w owns k tiles 2w, 2w + 1
    for nt in range_constexpr(4):
        ksl = range(2) if nt >= 2 else range(1)
        accs = []
        for kt in range_constexpr(2 * 0, 2):
            acc = Vec.filled(4, 0.0, fx.Float32)
            for ks in ksl:
                a = rd_c(R_KBT, 16 * (2 * role + kt) + cg, ks)
                b = rd_a(nt, ks)
                acc = mfma_bf16(a, b, acc)
            accs.append(acc)
        words = pack_bf16([zero - accs[0][e] for e in range_constexpr(4)] + [zero - accs[1][e] for e in range_constexpr(4)])
        buffer_ops.buffer_store(Vec.from_elements(words, fx.Int32), wsr, (REC_W + ((nt * 4 + role) * 64 + lane) * 16) // 4, cache_modifier=wcm)
    # u = A (beta v): wave w owns v tiles 2w, 2w + 1
    for vt2 in range_constexpr(2):
        for mt in range_constexpr(4):
            acc = Vec.filled(4, 0.0, fx.Float32)
            for ks in (range(2) if mt >= 2 else range(1)):
                a = rd_a(mt, ks)
                b = rd_c(R_VBT, 16 * (2 * role + vt2) + cg, ks)
                acc = mfma_bf16(a, b, acc)
            words = pack_bf16([acc[e] for e in range_constexpr(4)])
            buffer_ops.buffer_store(Vec.from_elements(words, fx.Int32), wsr, (REC_U + (((2 * role + vt2) * 4 + mt) * 64 + lane) * 8) // 4, cache_modifier=wcm)

    if const_expr(arg_flags is not None):
        fx.rocdl.s_waitcnt(vmcnt=0)
        barrier()
        if tx < WALK_GROUPS:
            fr = buffer_ops.create_buffer_resource_from_addr(arg_flags, num_records_bytes=fx.Int32(0x7FFFFFF0))
            buffer_ops.buffer_store(fx.Int32(1), fr, ((h * i32_nc + cid) * WALK_GROUPS + tx) * FLAG_STRIDE, cache_modifier=0x10)

    # ---- roll the conv state: the last 3 raw inputs of the sequence (chunk 0 only; history is in LDS)
    if first:
        for it in range_constexpr(2):
            chn = tx + it * 256
            band = chn // 128
            cc = chn % 128
            okc = (chn < 384) & first
            dim = band * LP + h * D + cc
            for p in range_constexpr(3):
                s = tn - 3 + p
                rv = buffer_ops.buffer_load(mr, (bos + s) * i32_ldx + dim, vec_width=1, dtype=fx.Int16, mask=(s >= 0) & okc)
                hp = fx.min(fx.max(s + 3, fx.Int32(0)), fx.Int32(2))
                hvv = fx.Int32(fx.Int16(lds_ld(base + R_HIST + ((band * 3 + hp) * 128 + cc) * 2, T.i16, 2)))
                val = (s >= 0).select(fx.Int32(rv) & 0xFFFF, hvv & 0xFFFF)
                buffer_ops.buffer_store(
                    raw(val.to(fx.Int16)), csr, dim * i32_cs_dim + p * i32_cs_pos, mask=okc
                )


@launcher
def compile_prologue():
    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name="kda_prefill_prologue_h12_d128", known_block_size=[256, 1, 1])
    def kernel(
        arg_mixed: fx.Int64,
        i32_ldx: fx.Int32,
        i32_mixed_bytes: fx.Int32,
        arg_rawg: fx.Int64,
        i32_ldg: fx.Int32,
        arg_beta: fx.Int64,
        i32_ldb: fx.Int32,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i32_cs_slot: fx.Int32,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_tab: fx.Int64,
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        f32_lb: fx.Float32,
        f32_scale: fx.Float32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        prologue_body(fx.Int32(fx.ptrtoint(smem.buf.ptr)), fx.Int32(fx.block_idx.x), fx.Int32(fx.block_idx.y), fx.Int32(fx.block_idx.x),
                      arg_mixed, i32_ldx, i32_mixed_bytes, arg_rawg, i32_ldg, arg_beta, i32_ldb, arg_alog, arg_dtb, arg_cw, arg_cs,
                      i32_cs_slot, i32_cs_dim, i32_cs_pos, arg_tab, arg_ws, i32_nc, f32_lb, f32_scale)

    @flyc.jit
    def launch(
        arg_mixed: fx.Int64,
        i32_ldx: fx.Int32,
        i32_mixed_bytes: fx.Int32,
        arg_rawg: fx.Int64,
        i32_ldg: fx.Int32,
        arg_beta: fx.Int64,
        i32_ldb: fx.Int32,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i32_cs_slot: fx.Int32,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_tab: fx.Int64,
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        f32_lb: fx.Float32,
        f32_scale: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(
            arg_mixed, i32_ldx, i32_mixed_bytes, arg_rawg, i32_ldg, arg_beta, i32_ldb, arg_alog, arg_dtb, arg_cw, arg_cs,
            i32_cs_slot, i32_cs_dim, i32_cs_pos, arg_tab, arg_ws, i32_nc, f32_lb, f32_scale,
        ).launch(grid=(fx.Int64(i32_nc), NH, 1), block=(256, 1, 1), stream=stream)

    return launch


@functools.cache
def _launcher():
    return compile_prologue()


def chunk_table(cu_seqlens, chunk_indices, state_indices, has_initial_state):
    """int32 [Nc, 4] per-chunk rows (sequence start, length, chunk start, slot | no-initial-state << 30); once per step."""
    seq = chunk_indices[:, 0].long()
    bos = cu_seqlens[seq]
    tn = cu_seqlens[seq + 1] - bos
    enc = state_indices[seq] | ((~has_initial_state[seq]).to(torch.int32) << 30)
    return torch.stack([bos, tn, chunk_indices[:, 1] * 64, enc], 1).to(torch.int32).contiguous()


def prologue(mixed, raw_g, beta_logit, A_log, dt_bias, lower_bound, scale, conv_weight, conv_state, state_indices,
             has_initial_state, cu_seqlens, chunk_indices, ws=None, table=None):
    """Workspace records [H, Nc, RECORD_BYTES] uint8 of the chunks of ``cu_seqlens``; rolls ``conv_state`` in place."""
    nc = chunk_indices.shape[0]
    assert mixed.dtype == raw_g.dtype == beta_logit.dtype == torch.bfloat16 and mixed.stride(-1) == raw_g.stride(-1) == beta_logit.stride(-1) == 1
    assert mixed.stride(0) % 8 == 0 and conv_state.stride(0) < 2**31 // 2
    if ws is None:
        ws = torch.empty(NH, nc, RECORD_BYTES, dtype=torch.uint8, device=mixed.device)
    cs = conv_state
    if table is None:
        table = chunk_table(cu_seqlens, chunk_indices, state_indices, has_initial_state)
    _run_compiled(
        _launcher(),
        mixed.data_ptr(), mixed.stride(0), mixed.shape[0] * mixed.stride(0) * 2,
        raw_g.data_ptr(), raw_g.stride(-3) if raw_g.dim() > 2 else raw_g.stride(0), beta_logit.data_ptr(), beta_logit.stride(-2),
        A_log.data_ptr(), dt_bias.data_ptr(), conv_weight.data_ptr(), cs.data_ptr(), cs.stride(0), cs.stride(1), cs.stride(2),
        table.data_ptr(), ws.data_ptr(), nc, float(lower_bound), float(scale), torch.cuda.current_stream(),
    )
    return ws
