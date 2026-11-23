# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Clamped SiLU-mul of a gate_up row (gate | up, 2 x 2304 columns) and the MXFP8 quant of the result, one workgroup of
576 lanes per row, 4 output columns per lane (a 32-column MX group is eight lanes, quad + half-row DPP amax).

Numerics of ``SiluAndMulWithClamp`` (default alpha 1, beta 0): ``silu(min(gate, limit))`` in fp32 rounded to bf16,
times ``clamp(up, -limit, limit) + 0`` (a -0 up reads as +0) in fp32 rounded to bf16; the quantizer then reads that bf16 value.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

I = 2304
NT = I // 4  # 576 lanes
TINY = 0x00800000  # fp32 tiny, the amax floor
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _dpp_f(v, ctrl):
    o = _llvm.mlir_undef(T.i32)
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, 0xF, 0xF, True)).bitcast(fx.Float32)


def _max_dpp(a, ctrl):
    return _maxnum(a, _dpp_f(a, ctrl))


def _unpack8(v):
    out = []
    for i in range(len(v)):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pk(a, b):
    """Two fp32 -> two bf16 in a dword (RNE), ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _set_fp16_ovfl():
    _llvm.call_intrinsic(
        None, "llvm.amdgcn.s.setreg", [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()], [], []
    )


def _cvt_pk_fp8(old, a, b, hi: bool):
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


def _e8m0(y):
    """(scale byte as fp32, 2^(127 - byte)) of the 32-column group whose 4 lanes hold ``y``."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = _maxnum(a, fx.math.absf(v))
    a = _max_dpp(a, 0x4E)
    a = _max_dpp(a, 0xB1)
    a = _maxnum(_max_dpp(a, 0x141), _f32(TINY))
    d = a / fx.Float32(448.0)
    small = d < _f32(TINY)
    lg = fx.Float32(
        _rocdl_ops.log(T.f32, fx.as_ir_value(small.select(d * fx.Float32(2.0**32), d)))
    ) - small.select(fx.Float32(32.0), fx.Float32(0.0))
    sb = _minnum(_maxnum(fx.math.ceil(lg) + fx.Float32(127.0), fx.Float32(0.0)), fx.Float32(254.0))
    rescale = fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.Float32(127.0) - sb)))
    return sb, rescale


def _codes(y, rescale):
    q = [v * rescale for v in y]
    return _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True)


def _xcd_row(bid, m):
    """Workgroup b runs on XCD b % 8, which takes a contiguous run of rows."""
    x, k = bid % 8, bid // 8
    q8, r8 = m // 8, m % 8
    extra = x < r8
    row = x * q8 + extra.select(x, r8) + k
    return row, k < q8 + extra.select(fx.Int32(1), fx.Int32(0))


def compile_clamp_silu_mul_mxfp8():
    name = f"clamp_silu_mul_mxfp8_i{I}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sx: fx.Int32,  # gate_up row stride (elements)
        arg_x: fx.Int64,  # [m, 2I] bf16
        arg_codes: fx.Int64,  # [m, I] e4m3
        arg_scale: fx.Int64,  # [m, I / 32] u8
        i32_limit: fx.Int32,  # fp32 bits
    ):
        _set_fp16_ovfl()
        tx = fx.thread_idx.x
        row, live = _xcd_row(fx.block_idx.x, i32_m)
        if live:
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(row) * fx.Int64(i32_sx * 2), num_records_bytes=4 * I
            )
            gv = _unpack8(Vec(buffer_ops.buffer_load(xr, tx * 2, vec_width=2, dtype=fx.Int32)))
            uv = _unpack8(Vec(buffer_ops.buffer_load(xr, I // 2 + tx * 2, vec_width=2, dtype=fx.Int32)))
            limit = i32_limit.bitcast(fx.Float32)
            g = [_minnum(v, limit) for v in gv]
            u = [_minnum(_maxnum(v, -limit), limit) + fx.Float32(0.0) for v in uv]
            sg = _unpack8([_pk(*[
                v / (fx.Float32(1.0) + fx.math.exp(-v)) for v in g[2 * i:2 * i + 2]
            ]) for i in range_constexpr(2)])
            pk = [_pk(sg[2 * i] * u[2 * i], sg[2 * i + 1] * u[2 * i + 1]) for i in range_constexpr(2)]
            y = _unpack8(pk)
            sb, rescale = _e8m0(y)
            codes = _codes(y, rescale)
            cr = buffer_ops.create_buffer_resource_from_addr(arg_codes, num_records_bytes=fx.Int64(i32_m) * I)
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_scale, num_records_bytes=fx.Int64(i32_m) * (I // 32)
            )
            buffer_ops.buffer_store(codes, cr, row * (I // 4) + tx)
            buffer_ops.buffer_store(
                fx.Int32(sb.to(fx.Int32)).to(fx.Int8),
                sr,
                row * (I // 32) + tx // 8,
                mask=(tx % 8) == 0,
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch_clamp_silu_mul_mxfp8(
        arg_x: fx.Int64,
        arg_codes: fx.Int64,
        arg_scale: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_limit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(i32_m, i32_sx, arg_x, arg_codes, arg_scale, i32_limit).launch(
            grid=(fx.Int64((i32_m + 7) // 8 * 8), 1, 1), block=(NT, 1, 1), stream=stream
        )

    launch_clamp_silu_mul_mxfp8.kernel_name = name
    launch_clamp_silu_mul_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_clamp_silu_mul_mxfp8
