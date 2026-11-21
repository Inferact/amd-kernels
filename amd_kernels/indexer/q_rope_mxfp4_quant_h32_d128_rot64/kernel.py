# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Indexer query GPT-J RoPE (last 64 of 128 dims) + MXFP4 quant and the scaled per-head weights, FlyDSL (gfx950).

A wave is 4 heads of one token, 16 lanes a head, 8 contiguous dims a lane (RoPE dims are lanes 8..15), so a 32-dim
MX group is 4 lanes and its amax a quad DPP. The rotated dims round through bf16 before the quant, as the bf16
tensor they stand for. Scale: e8m0 = ceil(log2(max(amax, 6 * 2^-126) * fl32(1 / 6))) from the exponent bits, values
times 2^(127 - e8m0 + 127) rounded to e2m1 (v_cvt_scalef32_pk_fp4_f32, ties to even), first of a pair in the low nibble.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

NW = 4  # waves per workgroup
H = 32
D = 128
ROT = 64
HPW = 4  # heads per wave
S = H // HPW  # waves per token
SIXTH = 0x3E2AAAAB  # fp32 1 / 6
FLOOR = 6 * 2.0**-126
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _dpp_f(v, ctrl):
    o = _llvm.mlir_undef(T.i32)
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, 0xF, 0xF, True)).bitcast(fx.Float32)


def _unpack8(v):
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _unpack4(v):
    out = []
    for i in range(2):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _rn_bf16_f32(x):
    """fp32 -> bf16 (RNE) -> fp32."""
    h = fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(x)), fx.as_ir_value(fx.Float32(0.0)))) & 0xFFFF
    return (h << 16).bitcast(fx.Float32)


def _e2m1x2(a, b, old, sel):
    return fx.Int32(
        _rocdl_ops.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _mx_scale(amax):
    """(e8m0 as i32, 2^(254 - e8m0)) of a group's amax."""
    ratio = _maxnum(amax, fx.Float32(FLOOR)) * fx.Int32(SIXTH).bitcast(fx.Float32)
    e = (_bits(ratio) + 0x7FFFFF) >> 23
    return e, (fx.Int32(254) - e << 23).bitcast(fx.Float32)


def compile_q_rope_mxfp4_quant(cs_bf16: bool):
    """``cs_bf16``: the cos | sin cache is bf16 (else fp32)."""
    name = f"q_rope_mxfp4_quant_h{H}_d{D}_rot{ROT}" + ("_csbf16" if cs_bf16 else "")
    CS_B = 2 if cs_bf16 else 4

    def _load_cs(cs_r, pos, jl):
        """cos and sin of the 4 RoPE pairs this lane rotates."""
        if cs_bf16:
            cw = Vec(buffer_ops.buffer_load(cs_r, pos * (ROT // 2) + jl * 2, vec_width=2, dtype=fx.Int32))
            sw = Vec(buffer_ops.buffer_load(cs_r, pos * (ROT // 2) + 16 + jl * 2, vec_width=2, dtype=fx.Int32))
            return _unpack4(cw), _unpack4(sw)
        cv = Vec(buffer_ops.buffer_load(cs_r, pos * ROT + jl * 4, vec_width=4, dtype=T.f32))
        sv = Vec(buffer_ops.buffer_load(cs_r, pos * ROT + 32 + jl * 4, vec_width=4, dtype=T.f32))
        return cv, sv

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_q: fx.Int64,  # [n, 32, 128] bf16
        arg_pos: fx.Int64,  # [n] i64
        arg_cs: fx.Int64,  # [rows, 64]: cos | sin
        arg_w: fx.Int64,  # [n, 32] bf16
        arg_qo: fx.Int64,  # [n, 32, 64] u8
        arg_so: fx.Int64,  # [n, 32, 4] u8
        arg_wo: fx.Int64,  # [n, 32] fp32
        i32_wscale: fx.Int32,  # fp32 bits
    ):
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        tok = wid // S
        w4 = wid % S
        if tok < i32_n:
            head = w4 * HPW + lane // 16
            lh = lane % 16
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_q + fx.Int64(tok) * (H * D * 2), num_records_bytes=H * D * 2
            )
            x = _unpack8(Vec(buffer_ops.buffer_load(xr, head * (D // 2) + lh * 4, vec_width=4, dtype=fx.Int32)))
            pos_r = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=fx.Int64(i32_n) * 8)
            pos = fx.Int32(buffer_ops.buffer_load(pos_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True))
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * CS_B)
            )
            jl = lh % 8
            cv, sv = _load_cs(cs_r, pos, jl)
            rope = lh >= 8
            y = []
            for i in range_constexpr(4):
                e, o = x[2 * i], x[2 * i + 1]
                y += [
                    rope.select(_rn_bf16_f32(e * cv[i] - o * sv[i]), e),
                    rope.select(_rn_bf16_f32(o * cv[i] + e * sv[i]), o),
                ]
            a = fx.math.absf(y[0])
            for v in y[1:]:
                a = _maxnum(a, fx.math.absf(v))
            a = _maxnum(a, _dpp_f(a, 0x4E))
            a = _maxnum(a, _dpp_f(a, 0xB1))
            e8, qs = _mx_scale(a)
            q = [v * qs for v in y]
            codes = _e2m1x2(q[6], q[7], _e2m1x2(q[4], q[5], _e2m1x2(q[2], q[3], _e2m1x2(q[0], q[1], 0, 0), 1), 2), 3)
            row = tok * H + head
            qo = buffer_ops.create_buffer_resource_from_addr(arg_qo, num_records_bytes=fx.Int64(i32_n) * (H * D // 2))
            buffer_ops.buffer_store(codes, qo, row * (D // 8) + lh)
            so = buffer_ops.create_buffer_resource_from_addr(arg_so, num_records_bytes=fx.Int64(i32_n) * (H * 4))
            buffer_ops.buffer_store(
                e8.to(fx.Int8), so, row * 4 + lh // 4, mask=(lh % 4) == 0, offset_is_bytes=True
            )
            if lane < HPW:
                wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=fx.Int64(i32_n) * (H * 2))
                wv = fx.Int32(buffer_ops.buffer_load(wr, tok * H + w4 * HPW + lane, vec_width=1, dtype=T.i16)).to(
                    fx.Int32
                )
                wf = (wv << 16).bitcast(fx.Float32) * i32_wscale.bitcast(fx.Float32)
                wo = buffer_ops.create_buffer_resource_from_addr(arg_wo, num_records_bytes=fx.Int64(i32_n) * (H * 4))
                buffer_ops.buffer_store(wf, wo, tok * H + w4 * HPW + lane)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_pos: fx.Int64,
        arg_cs: fx.Int64,
        arg_w: fx.Int64,
        arg_qo: fx.Int64,
        arg_so: fx.Int64,
        arg_wo: fx.Int64,
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        i32_wscale: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(i32_n, i32_cs_rows, arg_q, arg_pos, arg_cs, arg_w, arg_qo, arg_so, arg_wo, i32_wscale).launch(
            grid=(fx.Int64((i32_n * S + (NW - 1)) // NW), 1, 1), block=(64 * NW, 1, 1), stream=stream
        )

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
