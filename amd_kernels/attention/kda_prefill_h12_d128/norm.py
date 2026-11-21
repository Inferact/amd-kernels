# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gated RMSNorm of the KDA output rows (gfx950, FlyDSL): ``out = bf16(o * rsqrt(mean(o^2) + eps) * weight * sigmoid(gate))``
per (token, head) row of 128.

Sixteen lanes own a row (8 values each, one 16-byte load of ``o`` and of the gate row), a wave takes four rows per step and
unrolls ``unroll`` steps so every load of a workgroup is in flight before the first reduction; the row sum of squares is a
DPP reduction inside the 16-lane row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import D, NH

LOG2E = 1.4426950408889634


def _dpp(v, ctrl):
    o = _llvm.mlir_undef(T.i32)
    r = _rocdl_ods.update_dpp(T.i32, o, fx.as_ir_value(fx.Float32(v).bitcast(fx.Int32)), ctrl, 0xF, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def _sum16(v):
    for ctrl in (0xB1, 0x4E, 0x141, 0x140):
        v = v + _dpp(v, ctrl)
    return v


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.as_ir_value(fx.Float32(x))))


def _rcp(x):
    return fx.Float32(fx.rocdl.rcp(T.f32, fx.as_ir_value(fx.Float32(x))))


def _rsqrt(x):
    return fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(fx.Float32(x))))


def _bf16x8(v4):
    return [(fx.Int32(v4[i // 2]) << 16).bitcast(fx.Float32) if i % 2 == 0 else (fx.Int32(v4[i // 2]) & -65536).bitcast(fx.Float32)
            for i in range(8)]


@launcher
def compile_norm(waves: int, unroll: int, w_f32: bool, nt: int = 0):
    rows_per_cta = waves * unroll * 4
    name = f"kda_prefill_gated_norm_h12_d128_w{waves}_u{unroll}" + ("_wf32" if w_f32 else "") + (f"_nt{nt}" if nt else "")

    @flyc.kernel(name=name, known_block_size=[64 * waves, 1, 1])
    def kernel(
        arg_o: fx.Int64,
        arg_g: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        i32_gstride: fx.Int32,
        i32_rows: fx.Int32,
        i64_gbytes: fx.Int64,
        f32_eps: fx.Float32,
    ):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        r4 = lane // 16
        c8 = lane % 16
        o_r = buffer_ops.create_buffer_resource_from_addr(arg_o, num_records_bytes=i32_rows * (D * 2))
        out_r = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_rows * (D * 2))
        g_r = buffer_ops.create_buffer_resource_from_addr(arg_g, num_records_bytes=i64_gbytes)
        w_r = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=D * (4 if w_f32 else 2))
        if const_expr(w_f32):
            wv = []
            for h in range_constexpr(2):
                wv += list(Vec(buffer_ops.buffer_load(w_r, 8 * c8 + 4 * h, vec_width=4, dtype=fx.Float32)))
            wv = [fx.Float32(x) for x in wv]
        else:
            wv = _bf16x8(Vec(buffer_ops.buffer_load(w_r, 4 * c8, vec_width=4, dtype=fx.Int32)))
        row0 = (fx.block_idx.x * waves + wave) * (unroll * 4) + r4
        loads = []
        for u in range_constexpr(unroll):
            row = row0 + u * 4
            live = row < i32_rows
            tok = row // NH
            head = row - tok * NH
            ov = buffer_ops.buffer_load(o_r, row * (D // 2) + 4 * c8, vec_width=4, dtype=fx.Int32, mask=live, cache_modifier=nt)
            gv = buffer_ops.buffer_load(g_r, (tok * i32_gstride + head * D + 8 * c8) // 2, vec_width=4, dtype=fx.Int32, mask=live,
                                        cache_modifier=nt)
            loads.append((row, live, Vec(ov), Vec(gv)))
        for row, live, ov, gv in loads:
            x = _bf16x8(ov)
            g = _bf16x8(gv)
            ss = x[0] * x[0]
            for i in range_constexpr(1, 8):
                ss = ss + x[i] * x[i]
            rstd = _rsqrt(_sum16(ss) * (1.0 / D) + f32_eps)
            y = []
            for i in range_constexpr(8):
                sig = _rcp(fx.Float32(1.0) + _exp2(g[i] * (-LOG2E)))
                y.append(x[i] * rstd * wv[i] * sig)
            buffer_ops.buffer_store(Vec.from_elements(y, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32), out_r,
                                    row * (D // 2) + 4 * c8, mask=live)

    @flyc.jit
    def launch(
        arg_o: fx.Int64,
        arg_g: fx.Int64,
        arg_w: fx.Int64,
        arg_out: fx.Int64,
        i32_gstride: fx.Int32,
        i32_rows: fx.Int32,
        i64_gbytes: fx.Int64,
        f32_eps: fx.Float32,
        stream: fx.Stream,
    ):
        kernel(arg_o, arg_g, arg_w, arg_out, i32_gstride, i32_rows, i64_gbytes, f32_eps).launch(
            grid=((i32_rows + (rows_per_cta - 1)) // rows_per_cta, 1, 1), block=(64 * waves, 1, 1), stream=stream)

    return launch
