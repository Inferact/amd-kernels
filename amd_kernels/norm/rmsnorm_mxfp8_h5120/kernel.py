# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMSNorm of a 5120-column row to MXFP8 codes and row-major E8M0 scales in one launch: 320 lanes, 16 columns each.

A lane holds columns 8 l .. 8 l + 7 and 2560 + 8 l .. 2560 + 8 l + 7 (two 16-byte loads of x and of the weight), so a
32-column MX group is four consecutive lanes (a quad: two DPP max steps) and a wave's 64 lanes cover 8 groups of each
half. The square sum is a fixed tree: the lane's 16 squares, a wave butterfly, the five waves in wave order. Then
``y = bf16(x * rsqrt(mean + eps) * w)`` (one rounding, as the served RMSNorm op), quantized from that bf16 value: the
group's E8M0 byte and the e4m3 codes of ``y * 2^(127 - byte)``. Each output element has one writer.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

H = 5120
NT = H // 16  # 320 lanes
HALF = H // 2
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


def _dpp_i(v, ctrl, old=None, bank_mask=0xF):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(v), ctrl, 0xF, bank_mask, full))


def _swap(op, v):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(_llvm.extractvalue(T.i32, res, [1]))


def _xor_lane(v, m, lane):
    """i32 value of lane ``lane ^ m`` (m a power of two < 64)."""
    if m == 1:
        return _dpp_i(v, 0xB1)
    if m == 2:
        return _dpp_i(v, 0x4E)
    if m == 4:
        return _dpp_i(v, 0x104, old=_dpp_i(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    if m == 8:
        return _dpp_i(v, 0x128)
    a, b = _swap(_rocdl_ops.permlane16_swap if m == 16 else _rocdl_ops.permlane32_swap, v)
    return ((lane & m) != 0).select(a, b)


def _wave_sum(v, lane):
    """fp32 xor butterfly: every lane ends with the wave's sum, in a fixed order."""
    for m in (1, 2, 4, 8, 16, 32):
        v = v + _xor_lane(_bits(v), m, lane).bitcast(fx.Float32)
    return v


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
    """(scale byte as fp32, 2^(127 - byte)) of the 32-column group whose four lanes hold the 8 values ``y`` each."""
    a = fx.math.absf(y[0])
    for v in y[1:]:
        a = _maxnum(a, fx.math.absf(v))
    for ctrl in (0x4E, 0xB1):
        a = _maxnum(a, _dpp_i(_bits(a), ctrl).bitcast(fx.Float32))
    a = _maxnum(a, _f32(TINY))
    d = a / fx.Float32(448.0)
    small = d < _f32(TINY)
    lg = fx.Float32(
        _rocdl_ops.log(T.f32, fx.as_ir_value(small.select(d * fx.Float32(2.0**32), d)))
    ) - small.select(fx.Float32(32.0), fx.Float32(0.0))
    sb = _minnum(_maxnum(fx.math.ceil(lg) + fx.Float32(127.0), fx.Float32(0.0)), fx.Float32(254.0))
    rescale = fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.Float32(127.0) - sb)))
    return sb, rescale


def _codes(y, rescale):
    """Four fp32 -> four e4m3 in a dword."""
    q = [v * rescale for v in y]
    return _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True)


@launcher
def compile_rmsnorm_mxfp8():
    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 5]

    @flyc.kernel(name="rmsnorm_mxfp8_h5120", known_block_size=[NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [m, 5120] bf16, row stride i32_sx
        arg_w: fx.Int64,  # [5120] bf16
        arg_codes: fx.Int64,  # [m, 5120] e4m3
        arg_scale: fx.Int64,  # [m, 160] u8
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_eps: fx.Int32,  # fp32 bits
    ):
        _set_fp16_ovfl()
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        xr = buffer_ops.create_buffer_resource_from_addr(arg_x + fx.Int64(row) * fx.Int64(i32_sx * 2),
                                                         num_records_bytes=H * 2)
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=H * 2)
        xv, wv = [], []
        for h in range_constexpr(2):  # dword offset of 8 columns: (HALF * h + 8 tx) / 2
            off = HALF * h // 2 + tx * 4
            xv += _unpack8(Vec(buffer_ops.buffer_load(xr, off, vec_width=4, dtype=fx.Int32)))
            wv += _unpack8(Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32)))
        acc = fx.Float32(0.0)
        for i in range_constexpr(16):
            acc = fx.math.fma(xv[i], xv[i], acc)
        acc = _wave_sum(acc, lane)
        if lane == 0:
            part[wave] = acc
        fx.gpu.barrier()
        ss = (((part[0] + part[1]) + part[2]) + part[3]) + part[4]
        eps = i32_eps.bitcast(fx.Float32)
        rrms = fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(ss / fx.Float32(float(H)) + eps)))
        y = [(xv[i] * rrms) * wv[i] for i in range_constexpr(16)]
        pk = [_pk(y[2 * i], y[2 * i + 1]) for i in range_constexpr(8)]
        yb = _unpack8(pk)
        cr = buffer_ops.create_buffer_resource_from_addr(arg_codes + fx.Int64(row) * H, num_records_bytes=H)
        sr = buffer_ops.create_buffer_resource_from_addr(arg_scale + fx.Int64(row) * (H // 32),
                                                         num_records_bytes=H // 32)
        for h in range_constexpr(2):
            yh = yb[8 * h:8 * h + 8]
            sb, rescale = _e8m0(yh)
            codes = [_codes(yh[4 * j:4 * j + 4], rescale) for j in range_constexpr(2)]
            buffer_ops.buffer_store(Vec.from_elements(codes, fx.Int32), cr, (HALF * h) // 4 + tx * 2)
            buffer_ops.buffer_store(
                fx.Int32(sb.to(fx.Int32)).to(fx.Int8), sr, (HALF * h) // 32 + tx // 4, mask=(tx % 4) == 0,
                offset_is_bytes=True)

    @flyc.jit
    def launch_rmsnorm_mxfp8(arg_x: fx.Int64, arg_w: fx.Int64, arg_codes: fx.Int64, arg_scale: fx.Int64,
                             i32_m: fx.Int32, i32_sx: fx.Int32, i32_eps: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_w, arg_codes, arg_scale, i32_m, i32_sx, i32_eps).launch(
            grid=(fx.Int64(i32_m), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_rmsnorm_mxfp8.kernel_name = "rmsnorm_mxfp8_h5120"
    launch_rmsnorm_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_rmsnorm_mxfp8
