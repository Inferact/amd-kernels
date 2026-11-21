# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunk output (gfx950, FlyDSL): ``O = qg S^T + Aqk Vn`` of one (chunk, head) from the state record the walk published
(``S`` and ``Vn`` bf16 fragments) and the workspace record (``qg``, ``Aqk``), then the gated RMSNorm of the 64 x 128 tile.

Four waves, two 16-column value slices each: with ``S`` / ``Vn`` as the A operand, the 16 ``qg`` and the six causal ``Aqk`` fragments
as the B operand, a lane holds four consecutive value columns of one token (``O^T``). ``o`` is rounded to bf16, the row sum of squares
is reduced over the lanes of a token and the four waves through LDS, and ``out = bf16(o rsqrt(mean(o^2) + eps) w sigmoid(gate))``
is stored 8 bytes per lane; rows past the sequence end fall outside the output buffer.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
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
    bf16x2_hi,
    bf16x2_lo,
    lds_ptr,
    mfma,
    pk_bf16,
    raw,
    touch_args,
)
from .reference import NH, RECORD_AQK, RECORD_BYTES, RECORD_QG, D
from .scan import SC1, _fp
from .walk import SREC_BYTES, SREC_S, SREC_VN, flag_ready, spin_until

LOG2E = 1.4426950408889634
OUT_LDS = 4096


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.as_ir_value(fx.Float32(x))))


def _rcp(x):
    return fx.Float32(fx.rocdl.rcp(T.f32, fx.as_ir_value(fx.Float32(x))))


def _rsqrt(x):
    return fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(fx.Float32(x))))


@ASTRewriter.transform
def out_body(lds_base, cid, h, w_f32, arg_ws, i32_nc, arg_sb, arg_tab, arg_out, i32_ostride, arg_gate, i32_gstride, arg_w, f32_eps,
             arg_flags=None):
    """One workgroup's work: the output tile of chunk ``cid`` and head ``h``; with ``arg_flags`` it waits for the walk's state record."""
    tx = fx.thread_idx.x
    lane = tx % 64
    wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
    l16 = lane % 16
    qg = lane // 16
    if const_expr(arg_flags is not None):
        f_r = buffer_ops.create_buffer_resource_from_addr(arg_flags, num_records_bytes=fx.Int32(0x7FFFFFF0))
        f2 = NH * i32_nc * WALK_GROUPS * FLAG_STRIDE + ((h * i32_nc + cid) * WALK_GROUPS) * FLAG_STRIDE
        if wave == 0:
            for g_ in range_constexpr(WALK_GROUPS):
                spin_until(f_r, f2 + g_ * FLAG_STRIDE, tx >= 0, flag_ready(f_r, f2 + g_ * FLAG_STRIDE, tx >= 0), 8)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        barrier()
        if tx < WALK_GROUPS:
            buffer_ops.buffer_store(fx.Int32(0), f_r, f2 + tx * FLAG_STRIDE, cache_modifier=SC1)
    tab_r = buffer_ops.create_buffer_resource_from_addr(arg_tab, num_records_bytes=fx.Int32(0x7FFFFFF0))
    row = Vec(buffer_ops.buffer_load(tab_r, cid * 4, vec_width=4, is_scalar=True))
    bos, tn, t0 = fx.Int32(row[0]), fx.Int32(row[1]), fx.Int32(row[2])
    rows = fx.min(fx.Int32(64), tn - t0)
    tok0 = bos + t0
    rs = _fp(arg_ws + fx.Int64(h * i32_nc + cid) * RECORD_BYTES, RECORD_BYTES)
    sbase = arg_sb + fx.Int64(h * i32_nc + cid) * SREC_BYTES
    qgf = {}
    for ks in range_constexpr(4):
        for nt in range_constexpr(4):
            qgf[(nt, ks)] = Vec(buffer_ops.buffer_load(rs, (RECORD_QG + (nt * 4 + ks) * 1024) // 4 + lane * 4, vec_width=4, dtype=fx.Int32))
    aqf = {}
    for ksp in range_constexpr(2):
        for nt in range_constexpr(2 * ksp, 4):
            aqf[(nt, ksp)] = Vec(buffer_ops.buffer_load(rs, (RECORD_AQK + (nt * 2 + ksp) * 1024) // 4 + lane * 4, vec_width=4, dtype=fx.Int32))
    sfv, vnv = [], []
    for vt in range_constexpr(2):
        vs = 2 * wave + vt
        sr = _fp(sbase + (SREC_S + vs * 4096), 4096)
        sfv.append([Vec(buffer_ops.buffer_load(sr, (ks * 1024) // 4 + lane * 4, vec_width=4, dtype=fx.Int32)) for ks in range_constexpr(4)])
        vr = _fp(sbase + (SREC_VN + vs * 2048), 2048)
        vnv.append([Vec(buffer_ops.buffer_load(vr, (ksp * 1024) // 4 + lane * 4, vec_width=4, dtype=fx.Int32)) for ksp in range_constexpr(2)])
    gr = _fp(arg_gate + fx.Int64(tok0) * (i32_gstride * 2), rows * (i32_gstride * 2))
    out_r = _fp(arg_out + fx.Int64(tok0) * (i32_ostride * 2), rows * (i32_ostride * 2))
    w_r = _fp(arg_w, D * (4 if w_f32 else 2))
    gate, wt = [], []
    for vt in range_constexpr(2):
        vs = 2 * wave + vt
        col = 16 * vs + 4 * qg
        gate.append([Vec(buffer_ops.buffer_load(gr, ((16 * nt + l16) * i32_gstride + h * D + col) // 2, vec_width=2, dtype=fx.Int32))
                     for nt in range_constexpr(4)])
        if const_expr(w_f32):
            wt.append(Vec(buffer_ops.buffer_load(w_r, col, vec_width=4, dtype=fx.Float32)))
        else:
            w2 = Vec(buffer_ops.buffer_load(w_r, col // 2, vec_width=2, dtype=fx.Int32))
            wt.append(Vec.from_elements([bf16x2_lo(w2[0]), bf16x2_hi(w2[0]), bf16x2_lo(w2[1]), bf16x2_hi(w2[1])], fx.Float32))

    ob = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)] for _ in range_constexpr(2)]
    for ks in range_constexpr(4):
        for nt in range_constexpr(4):
            for vt in range_constexpr(2):
                ob[vt][nt] = mfma(sfv[vt][ks], qgf[(nt, ks)], ob[vt][nt])
    for ksp in range_constexpr(2):
        for nt in range_constexpr(2 * ksp, 4):
            for vt in range_constexpr(2):
                ob[vt][nt] = mfma(vnv[vt][ksp], aqf[(nt, ksp)], ob[vt][nt])
    ow = []
    for vt in range_constexpr(2):
        row_w = []
        for nt in range_constexpr(4):
            o = ob[vt][nt]
            w2 = o.to(fx.BFloat16).bitcast(fx.Int32)
            row_w.append([fx.Int32(w2[0]), fx.Int32(w2[1])])
        ow.append(row_w)
    part = []
    for nt in range_constexpr(4):
        ss = fx.Float32(0.0)
        for vt in range_constexpr(2):
            for k in range_constexpr(2):
                a, b = bf16x2_lo(ow[vt][nt][k]), bf16x2_hi(ow[vt][nt][k])
                ss = ss + a * a + b * b
        part.append(ss)
    pa = lds_base + ((wave * 4 + qg) * 64 + l16) * 4
    for nt in range_constexpr(4):
        llvm.store(raw(part[nt]), lds_ptr(pa + nt * 64), alignment=4)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    barrier()
    rstd = []
    for nt in range_constexpr(4):
        tot = fx.Float32(0.0)
        for g in range_constexpr(16):
            tot = tot + fx.Float32(llvm.load(T.f32, lds_ptr(lds_base + (g * 64 + nt * 16 + l16) * 4), alignment=4))
        rstd.append(_rsqrt(tot * (1.0 / D) + f32_eps))
    for vt in range_constexpr(2):
        vs = 2 * wave + vt
        for nt in range_constexpr(4):
            ov = [bf16x2_lo(ow[vt][nt][0]), bf16x2_hi(ow[vt][nt][0]), bf16x2_lo(ow[vt][nt][1]), bf16x2_hi(ow[vt][nt][1])]
            g4 = [bf16x2_lo(gate[vt][nt][0]), bf16x2_hi(gate[vt][nt][0]), bf16x2_lo(gate[vt][nt][1]), bf16x2_hi(gate[vt][nt][1])]
            y = []
            for e in range_constexpr(4):
                sig = _rcp(fx.Float32(1.0) + _exp2(g4[e] * (-LOG2E)))
                y.append(ov[e] * rstd[nt] * wt[vt][e] * sig)
            words = Vec.from_elements([pk_bf16(y[0], y[1]), pk_bf16(y[2], y[3])], fx.Int32)
            buffer_ops.buffer_store(words, out_r, ((16 * nt + l16) * i32_ostride + h * D + 16 * vs + 4 * qg) // 2)


@launcher
def compile_out(w_f32: bool):
    name = "kda_prefill_out_h12_d128" + ("_wf32" if w_f32 else "")

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, OUT_LDS, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_sb: fx.Int64,
        arg_tab: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        arg_gate: fx.Int64,
        i32_gstride: fx.Int32,
        arg_w: fx.Int64,
        f32_eps: fx.Float32,
    ):
        touch_args(arg_ws, i32_nc, arg_sb, arg_tab, arg_out, i32_ostride, arg_gate, i32_gstride, arg_w, f32_eps)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        out_body(fx.Int32(fx.ptrtoint(smem.buf.ptr)), fx.Int32(fx.block_idx.x), fx.Int32(fx.block_idx.y), w_f32, arg_ws, i32_nc, arg_sb,
                 arg_tab, arg_out, i32_ostride, arg_gate, i32_gstride, arg_w, f32_eps)

    @flyc.jit
    def launch(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_sb: fx.Int64,
        arg_tab: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        arg_gate: fx.Int64,
        i32_gstride: fx.Int32,
        arg_w: fx.Int64,
        f32_eps: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(arg_ws, i32_nc, arg_sb, arg_tab, arg_out, i32_ostride, arg_gate, i32_gstride, arg_w, f32_eps).launch(
            grid=(i32_nc, NH, 1), block=(256, 1, 1), stream=stream)

    return launch
