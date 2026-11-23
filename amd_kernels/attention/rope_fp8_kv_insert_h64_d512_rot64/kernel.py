# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Query RoPE + e4m3 quant and KV RoPE + e4m3 cache insert, FlyDSL (gfx950): a wave is one head (64 query heads) or the
KV row (65th wave) of a token, 8 contiguous dims per lane; the 64 RoPE dims (GPT-J pairs 448..511) are lanes 56..63.

Values stay fp32 from the bf16 input through RoPE (even: fma(e, cos, -(o * sin)), odd: fma(o, cos, e * sin), the
contraction of the C op the kernel replaces) to the e4m3 conversion:
q * q_scale_inv for queries, kv * (1 / kv_scale) for the cache row, both clamped to +-448.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

NW = 4  # waves per workgroup
H = 64
D = 512
ROT = 64
ROW_B = 2 * D  # bytes of a bf16 row
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _unpack8(v):
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _set_fp16_ovfl():
    _llvm.call_intrinsic(
        None, "llvm.amdgcn.s.setreg", [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()], [], []
    )


def _cvt_pk_fp8(old, a, b, hi: bool):
    """Two fp32 -> e4m3 bytes (v_cvt_scalef32_pk_fp8_f32, scale 1) into the low or high half of ``old``."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _codes(y):
    """Eight fp32 -> two dwords of e4m3 bytes."""
    lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), y[0], y[1], False), y[2], y[3], True)
    hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), y[4], y[5], False), y[6], y[7], True)
    return [lo, hi]


def compile_rope_fp8_kv_insert(block_size: int):
    """``block_size``: cache rows per page (cache ``[pages, block_size, 512]`` e4m3)."""
    S = H + 1
    BS = block_size
    name = f"rope_fp8_kv_insert_h{H}_d{D}_rot{ROT}_p{BS}"

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        arg_q: fx.Int64,  # [n, 64, 512] bf16
        arg_kv: fx.Int64,  # [n, 512] bf16
        arg_qo: fx.Int64,  # [n, 64, 512] e4m3
        arg_cache: fx.Int64,  # [pages, BS, 512] e4m3
        arg_slot: fx.Int64,  # [n] i64, < 0 skips the KV insert
        arg_pos: fx.Int64,  # [n] i64
        arg_cs: fx.Int64,  # [rows, 64] fp32: cos | sin
        arg_ks: fx.Int64,  # kv scale, fp32 scalar
        arg_qsi: fx.Int64,  # q scale inverse, fp32 scalar
        i64_page_b: fx.Int64,  # page stride (bytes)
        i32_nsl: fx.Int32,  # slots given: tokens >= i32_nsl are not inserted
    ):
        _set_fp16_ovfl()
        tx, bx = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        wid = bx * NW + wave
        tok = wid // S
        s = wid % S
        if tok < i32_n:
            is_kv = s == H
            src = is_kv.select(
                arg_kv + fx.Int64(tok) * ROW_B, arg_q + (fx.Int64(tok) * H + fx.Int64(s)) * ROW_B
            )
            xr = buffer_ops.create_buffer_resource_from_addr(src, num_records_bytes=ROW_B)
            x = _unpack8(Vec(buffer_ops.buffer_load(xr, lane * 4, vec_width=4, dtype=fx.Int32)))
            pos_r = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=fx.Int64(i32_n) * 8)
            pos = fx.Int32(buffer_ops.buffer_load(pos_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True))
            cs_r = buffer_ops.create_buffer_resource_from_addr(
                arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * 4)
            )
            jl = lane % 8
            cv = Vec(buffer_ops.buffer_load(cs_r, pos * ROT + jl * 4, vec_width=4, dtype=T.f32))
            sv = Vec(buffer_ops.buffer_load(cs_r, pos * ROT + 32 + jl * 4, vec_width=4, dtype=T.f32))
            rope = lane >= 56
            y = []
            for i in range_constexpr(4):
                e, o = x[2 * i], x[2 * i + 1]
                y += [
                    rope.select(fx.math.fma(e, cv[i], -(o * sv[i])), e),
                    rope.select(fx.math.fma(o, cv[i], e * sv[i]), o),
                ]
            sc_r = buffer_ops.create_buffer_resource_from_addr(
                is_kv.select(arg_ks, arg_qsi), num_records_bytes=4
            )
            sc = fx.Int32(buffer_ops.buffer_load(sc_r, 0, vec_width=1, dtype=fx.Int32, is_scalar=True)).bitcast(fx.Float32)
            mul = is_kv.select(fx.Float32(1.0) / sc, sc)
            lim = fx.Float32(448.0)
            y = [_minnum(_maxnum(v * mul, -lim), lim) for v in y]
            codes = Vec.from_elements(_codes(y), fx.Int32)
            if is_kv:
                sl_r = buffer_ops.create_buffer_resource_from_addr(arg_slot, num_records_bytes=fx.Int64(i32_n) * 8)
                slot = fx.Int32(buffer_ops.buffer_load(sl_r, tok * 2, vec_width=1, dtype=fx.Int32, is_scalar=True))
                rec = (
                    fx.Int64(arg_cache)
                    + fx.Int64(slot // BS) * i64_page_b
                    + fx.Int64(slot % BS) * D
                )
                cr = buffer_ops.create_buffer_resource_from_addr(rec, num_records_bytes=D)
                buffer_ops.buffer_store(codes, cr, lane * 2, mask=(slot >= 0) & (tok < i32_nsl))
            else:
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_qo + (fx.Int64(tok) * H + fx.Int64(s)) * D, num_records_bytes=D
                )
                buffer_ops.buffer_store(codes, qr, lane * 2)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_qo: fx.Int64,
        arg_cache: fx.Int64,
        arg_slot: fx.Int64,
        arg_pos: fx.Int64,
        arg_cs: fx.Int64,
        arg_ks: fx.Int64,
        arg_qsi: fx.Int64,
        i32_n: fx.Int32,
        i32_cs_rows: fx.Int32,
        i64_page_b: fx.Int64,
        i32_nsl: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_n, i32_cs_rows, arg_q, arg_kv, arg_qo, arg_cache, arg_slot, arg_pos, arg_cs, arg_ks, arg_qsi,
            i64_page_b, i32_nsl,
        ).launch(grid=(fx.Int64((i32_n * S + (NW - 1)) // NW), 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
