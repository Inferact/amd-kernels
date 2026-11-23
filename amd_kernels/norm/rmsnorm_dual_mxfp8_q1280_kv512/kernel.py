# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMSNorm of a 1280-column q row to MXFP8 and of a 512-column kv row to bf16 in one launch, 7 waves per row.

Waves 0-4 hold the q row (4 columns per lane; a 32-column MX group is eight lanes, quad + half-row DPP amax), waves 5-6
the kv row. Both paths run the same instructions; the loads, stores and divisor are selected by the wave, so one barrier
joins the seven wave sums. The square sum follows the served Triton norm (8 warps of 4 columns per thread): the 4-column
tree, the DPP wave fold of ``tl.sum`` (valid in lane 63) and its warp tree, so the fp32 variance matches bit for bit;
then ``x * rsqrt(variance + eps) * w`` rounded to bf16, which the quantizer reads.
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

Q, KV = 1280, 512
NQ = Q // 4  # 320 q lanes
NKV = KV // 4  # 128 kv lanes
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


def _dpp_i(v, ctrl, row_mask=0xF, old=None):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(v), ctrl, row_mask, 0xF, True))


def _dpp_f(v, ctrl, row_mask=0xF, old=None):
    return _dpp_i(_bits(v), ctrl, row_mask, None if old is None else _bits(old)).bitcast(fx.Float32)


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


def _wave_sum_lane63(acc):
    """Row_shr 8/4/2/1 adds, row_bcast 15 / 31: the wave sum is valid in lane 63."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f(acc, 0x110 + sh)
    acc = _dpp_f(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f(acc, 0x143)


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


def compile_rmsnorm_dual_mxfp8():
    name = "rmsnorm_dual_mxfp8_q1280_kv512"

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 7]

    @flyc.kernel(name=name, known_block_size=[448, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sq: fx.Int32,  # q row stride (elements)
        i32_skv: fx.Int32,  # kv row stride
        arg_q: fx.Int64,  # [m, 1280] bf16
        arg_kv: fx.Int64,  # [m, 512] bf16
        arg_wq: fx.Int64,  # [1280] bf16
        arg_wkv: fx.Int64,  # [512] bf16
        arg_codes: fx.Int64,  # [m, 1280] e4m3
        arg_scale: fx.Int64,  # [m, 40] u8
        arg_kvout: fx.Int64,  # [m, 512] bf16
        i32_eps: fx.Int32,  # fp32 bits
    ):
        _set_fp16_ovfl()
        tx = fx.thread_idx.x
        row, live = _xcd_row(fx.block_idx.x, i32_m)
        if live:
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            is_kv = wave >= 5
            q_act = tx < NQ
            kv_act = (tx >= NQ) & (tx < NQ + NKV)
            kt = tx - NQ
            smem = fx.SharedAllocator().allocate(Shared).peek()
            part = smem.part.ptr
            qr = buffer_ops.create_buffer_resource_from_addr(
                arg_q + fx.Int64(row) * fx.Int64(i32_sq * 2), num_records_bytes=Q * 2
            )
            kr = buffer_ops.create_buffer_resource_from_addr(
                arg_kv + fx.Int64(row) * fx.Int64(i32_skv * 2), num_records_bytes=KV * 2
            )
            wqr = buffer_ops.create_buffer_resource_from_addr(arg_wq, num_records_bytes=Q * 2)
            wkr = buffer_ops.create_buffer_resource_from_addr(arg_wkv, num_records_bytes=KV * 2)
            xq = Vec(buffer_ops.buffer_load(qr, tx * 2, vec_width=2, dtype=fx.Int32, mask=q_act))
            xk = Vec(buffer_ops.buffer_load(kr, kt * 2, vec_width=2, dtype=fx.Int32, mask=kv_act))
            wq = Vec(buffer_ops.buffer_load(wqr, tx * 2, vec_width=2, dtype=fx.Int32, mask=q_act))
            wk = Vec(buffer_ops.buffer_load(wkr, kt * 2, vec_width=2, dtype=fx.Int32, mask=kv_act))
            xv = _unpack8([fx.Int32(xq[i]) | fx.Int32(xk[i]) for i in range_constexpr(2)])
            wv = _unpack8([fx.Int32(wq[i]) | fx.Int32(wk[i]) for i in range_constexpr(2)])
            a0 = fx.math.fma(xv[0], xv[0], xv[2] * xv[2])
            a1 = fx.math.fma(xv[1], xv[1], xv[3] * xv[3])
            acc = _wave_sum_lane63(a0 + a1)
            if lane == 63:
                part[wave] = acc
            fx.gpu.barrier()
            ss = is_kv.select(part[5] + part[6], ((part[0] + part[4]) + part[2]) + (part[1] + part[3]))
            n = is_kv.select(fx.Float32(float(KV)), fx.Float32(float(Q)))
            eps = i32_eps.bitcast(fx.Float32)
            rrms = fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(ss / n + eps)))
            y = [(xv[i] * rrms) * wv[i] for i in range_constexpr(4)]
            pk = [_pk(y[2 * i], y[2 * i + 1]) for i in range_constexpr(2)]
            yb = _unpack8(pk)
            kvo = buffer_ops.create_buffer_resource_from_addr(
                arg_kvout, num_records_bytes=fx.Int64(i32_m) * (KV * 2)
            )
            buffer_ops.buffer_store(Vec.from_elements(pk, fx.Int32), kvo, row * (KV // 2) + kt * 2, mask=kv_act)
            sb, rescale = _e8m0(yb)
            codes = _codes(yb, rescale)
            cr = buffer_ops.create_buffer_resource_from_addr(
                arg_codes, num_records_bytes=fx.Int64(i32_m) * Q
            )
            sr = buffer_ops.create_buffer_resource_from_addr(
                arg_scale, num_records_bytes=fx.Int64(i32_m) * (Q // 32)
            )
            buffer_ops.buffer_store(codes, cr, row * (Q // 4) + tx, mask=q_act)
            buffer_ops.buffer_store(
                fx.Int32(sb.to(fx.Int32)).to(fx.Int8),
                sr,
                row * (Q // 32) + tx // 8,
                mask=q_act & ((tx % 8) == 0),
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch_rmsnorm_dual_mxfp8(
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_wq: fx.Int64,
        arg_wkv: fx.Int64,
        arg_codes: fx.Int64,
        arg_scale: fx.Int64,
        arg_kvout: fx.Int64,
        i32_m: fx.Int32,
        i32_sq: fx.Int32,
        i32_skv: fx.Int32,
        i32_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, i32_sq, i32_skv, arg_q, arg_kv, arg_wq, arg_wkv, arg_codes, arg_scale, arg_kvout, i32_eps
        ).launch(grid=(fx.Int64((i32_m + 7) // 8 * 8), 1, 1), block=(448, 1, 1), stream=stream)

    launch_rmsnorm_dual_mxfp8.kernel_name = name
    launch_rmsnorm_dual_mxfp8.compile_hints = PRELOAD_HINTS
    return launch_rmsnorm_dual_mxfp8
