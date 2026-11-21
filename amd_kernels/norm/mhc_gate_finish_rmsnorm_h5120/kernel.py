# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""mHC seam finisher: tile partials -> gates (Sinkhorn) + RMSNorm of the collapse (+ MXFP8), one row per workgroup.

The producer GEMM (``gemm/mxfp8_postmix_hc4_n5120_k8192``) has already written the new residual, the collapse ``x1``
(bf16) and, per (row, column block), a 32-float record: the 24 gate-projection sums over the block's columns of all four
streams, the sum of squares of the residual (slot 24) and of ``x1`` (slot 25). This kernel never reads the residual: 640
lanes (10 waves) load the row's ``NB`` records at once (a lane takes one float4 of each of its tiles, in tile order), fold
them over lanes by DPP / permlane swaps and over the waves through LDS in a fixed order (bitwise repeatable), and then
split: every lane normalizes its 8 columns of ``x1`` (with the MXFP8 quantization of the result, 4 lanes per 32-column
block) while wave 0 turns the sums into the next seam's gates exactly as the seam kernel does (``rsqrt`` of the residual
mean square, scale and base, sigmoid, 4 x 4 Sinkhorn in the 16 lanes of a row).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

H = 5120
NS = 4
ROW_DW = H // 2
NTHR = H // 8
WAVES = NTHR // 64
PSLOT = 32
LOG2E = 1.4426950408889634
DPP_QUAD_SWAP1, DPP_QUAD_SWAP2 = 0xB1, 0x4E
DPP_ROW_ROR4, DPP_ROW_ROR8 = 0x124, 0x128
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _dpp_f(v, ctrl):
    o = _llvm.mlir_undef(T.i32)
    r = _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, 0xF, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def _maxnum(a, b):
    return fx.Float32(fx.arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _rcp(x):
    return fx.Float32(_rocdl_ops.rcp(T.f32, fx.as_ir_value(fx.Float32(x))))


def _exp2(x):
    return fx.Float32(_rocdl_ops.exp2(T.f32, fx.as_ir_value(fx.Float32(x))))


def _rsq(x):
    return fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(fx.Float32(x))))


def _unpack8(v):
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pack8(vals):
    return [
        fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(vals[2 * i])),
                                          fx.as_ir_value(fx.Float32(vals[2 * i + 1]))))
        for i in range(4)
    ]


def _swap_sum(x, op):
    """x + the value of the lane ``op`` pairs it with (a permlane16 / permlane32 swap)."""
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(_bits(x)), fx.as_ir_value(_bits(x)), False, False)
    return _f32(_llvm.extractvalue(T.i32, res, [0])) + _f32(_llvm.extractvalue(T.i32, res, [1]))


def _lane_group_sum(x):
    """Sum over the 8 lane groups of a wave (lanes l, l ^ 8, l ^ 16, l ^ 32 ...), in every lane."""
    x = x + _dpp_f(x, DPP_ROW_ROR8)
    x = _swap_sum(x, _rocdl_ops.permlane16_swap)
    return _swap_sum(x, _rocdl_ops.permlane32_swap)


def _codes(yb, rescale):
    """e4m3 codes of 8 values times ``rescale`` as two dwords."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))

    def cvt(old, a, b, hi):
        res = _llvm.call_intrinsic(
            v2i16, "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
            [_llvm.bitcast(v2i16, fx.as_ir_value(old)), fx.as_ir_value(a), fx.as_ir_value(b),
             fx.as_ir_value(fx.Float32(1.0)), fx.Boolean(hi).ir_value()], [], [],
        )  # fmt: skip
        return fx.Int32(_llvm.bitcast(T.i32, res))

    q = [v * rescale for v in yb]
    return [cvt(cvt(fx.Int32(0), q[2 * i], q[2 * i + 1], False), q[2 * i + 2], q[2 * i + 3], True) for i in (0, 2)]


def _e8m0_quad(yb):
    """(byte as i32, 2^(127 - byte)) of the 32-column group held by 4 consecutive lanes."""
    a = fx.math.absf(yb[0])
    for v in yb[1:]:
        a = _maxnum(a, fx.math.absf(v))
    a = _maxnum(a, _dpp_f(a, DPP_QUAD_SWAP1))
    a = _maxnum(a, _dpp_f(a, DPP_QUAD_SWAP2))
    a = _maxnum(a, _f32(0x00800000))
    bits = (a / fx.Float32(448.0)).bitcast(fx.Int32)
    up = (bits + 0x7FFFFF) >> 23
    ex = (bits <= 0x00400000).select(fx.Int32(0), (up < 1).select(fx.Int32(1), (up > 254).select(fx.Int32(254), up)))
    return ex, ((254 - ex) << 23).bitcast(fx.Float32)


def compile_mhc_gate_finish(nb, quant, repeat):
    tiles_per = -(-nb // (NTHR // 8))  # tiles a lane sums (lane tile = tx // 8 + 80 * i)
    name = f"mhc_gate_finish_h5120_nb{nb}" + ("_mx" if quant else "")

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, WAVES * PSLOT]
        tot: fx.Array[fx.Float32, PSLOT]

    @flyc.kernel(name=name, known_block_size=[NTHR, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_rms_eps: fx.Int32,
        i32_pre_eps: fx.Int32,
        i32_sk_eps: fx.Int32,
        i32_post_mult: fx.Int32,
        i32_norm_eps: fx.Int32,
        arg_gp: fx.Int64,  # [m, nb, 32] f32
        arg_x1: fx.Int64,  # [m, H] bf16
        arg_scale: fx.Int64,  # [3] f32
        arg_base: fx.Int64,  # [24] f32
        arg_nw: fx.Int64,  # [H] bf16
        arg_post: fx.Int64,  # [m, 4] f32
        arg_comb: fx.Int64,  # [m, 16] f32
        arg_npre: fx.Int64,  # [m, 4] f32
        arg_out: fx.Int64,  # [m, H] bf16
        arg_codes: fx.Int64,  # [m, H] e4m3
        arg_qs: fx.Int64,  # [m, H / 32] u8
    ):
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part, tot = smem.part.ptr, smem.tot.ptr
        m64 = fx.Int64(i32_m)

        def rsrc(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)

        def ldf(r, off, n):
            v = buffer_ops.buffer_load(r, off, vec_width=n, dtype=fx.Float32)
            return Vec(v) if n > 1 else fx.Float32(v)

        gpr = rsrc(arg_gp, m64 * (nb * PSLOT * 4))
        chunk, g = tx % 8, tx // 8
        recs = [
            Vec(buffer_ops.buffer_load(gpr, (row * nb + g + (NTHR // 8) * i) * PSLOT + chunk * 4, vec_width=4,
                                       dtype=fx.Float32, mask=g + (NTHR // 8) * i < nb))
            for i in range_constexpr(tiles_per)
        ]
        x1v = _unpack8(Vec(buffer_ops.buffer_load(rsrc(arg_x1, m64 * (H * 2)), row * ROW_DW + tx * 4, vec_width=4,
                                                  dtype=fx.Int32)))
        w = _unpack8(Vec(buffer_ops.buffer_load(rsrc(arg_nw, H * 2), tx * 4, vec_width=4, dtype=fx.Int32)))
        acc = recs[0]
        for r in recs[1:]:
            acc = acc + r
        acc = Vec.from_elements([_lane_group_sum(acc[k]) for k in range(4)], fx.Float32)
        if lane < 8:
            for k in range_constexpr(4):
                part[wave * PSLOT + lane * 4 + k] = acc[k]
        gpu.barrier()
        if tx < PSLOT:
            v = part[tx]
            for wv in range_constexpr(1, WAVES):
                v = v + part[wv * PSLOT + tx]
            tot[tx] = v
        gpu.barrier()
        ssx = tot[25]
        rstd = _rsq(ssx * fx.Float32(1.0 / H) + i32_norm_eps.bitcast(fx.Float32))
        yp = _pack8([(x1v[e] * rstd) * w[e] for e in range_constexpr(8)])
        buffer_ops.buffer_store(Vec.from_elements(yp, fx.Int32), rsrc(arg_out, m64 * (H * 2)), row * ROW_DW + tx * 4)
        if fx.const_expr(quant):
            yb = _unpack8(yp)
            ex, rescale = _e8m0_quad(yb)
            cd = _codes(yb, rescale)
            buffer_ops.buffer_store(Vec.from_elements(cd, fx.Int32), rsrc(arg_codes, m64 * H), row * (H // 4) + tx * 2)
            buffer_ops.buffer_store(ex.to(fx.Int8), rsrc(arg_qs, m64 * (H // 32)), row * (H // 32) + tx // 4,
                                    mask=tx % 4 == 0, offset_is_bytes=True)
        if wave == 0:
            e = lane % 16
            s0 = ldf(rsrc(arg_scale, 12), 0, 1)
            s1 = ldf(rsrc(arg_scale, 12), 1, 1)
            s2 = ldf(rsrc(arg_scale, 12), 2, 1)
            br = rsrc(arg_base, 96)
            sk_eps = i32_sk_eps.bitcast(fx.Float32)
            rs = _rsq(tot[24] * fx.Float32(1.0 / (NS * H)) + i32_rms_eps.bitcast(fx.Float32))
            a = tot[8 + e] * rs * s2 + ldf(br, 8 + e, 1)
            mx = _maxnum(a, _dpp_f(a, DPP_QUAD_SWAP1))
            mx = _maxnum(mx, _dpp_f(mx, DPP_QUAD_SWAP2))
            ex2 = _exp2((a - mx) * fx.Float32(LOG2E))
            sm = ex2 + _dpp_f(ex2, DPP_QUAD_SWAP1)
            sm = sm + _dpp_f(sm, DPP_QUAD_SWAP2)
            pm = ex2 * _rcp(sm) + sk_eps
            cs = pm + _dpp_f(pm, DPP_ROW_ROR4)
            cs = cs + _dpp_f(cs, DPP_ROW_ROR8)
            pm = pm * _rcp(cs + sk_eps)
            for _ in range_constexpr(repeat - 1):
                sm = pm + _dpp_f(pm, DPP_QUAD_SWAP1)
                sm = sm + _dpp_f(sm, DPP_QUAD_SWAP2)
                pm = pm * _rcp(sm + sk_eps)
                cs = pm + _dpp_f(pm, DPP_ROW_ROR4)
                cs = cs + _dpp_f(cs, DPP_ROW_ROR8)
                pm = pm * _rcp(cs + sk_eps)
            buffer_ops.buffer_store(pm, rsrc(arg_comb, m64 * 64), row * 16 + e, mask=lane < 16)
            ep = e % 8
            v = tot[ep] * rs * (e < 4).select(s0, s1) + ldf(br, ep, 1)
            sg = _rcp(fx.Float32(1.0) + _exp2(v * fx.Float32(-LOG2E)))
            buffer_ops.buffer_store(sg + i32_pre_eps.bitcast(fx.Float32), rsrc(arg_npre, m64 * 16), row * 4 + e,
                                    mask=(e < 4) & (lane < 16))
            buffer_ops.buffer_store(sg * i32_post_mult.bitcast(fx.Float32), rsrc(arg_post, m64 * 16), row * 4 + e - 4,
                                    mask=(e >= 4) & (e < 8) & (lane < 16))

    @flyc.jit
    def launch(
        arg_gp: fx.Int64,
        arg_x1: fx.Int64,
        arg_scale: fx.Int64,
        arg_base: fx.Int64,
        arg_nw: fx.Int64,
        arg_post: fx.Int64,
        arg_comb: fx.Int64,
        arg_npre: fx.Int64,
        arg_out: fx.Int64,
        arg_codes: fx.Int64,
        arg_qs: fx.Int64,
        i32_m: fx.Int32,
        i32_rms_eps: fx.Int32,
        i32_pre_eps: fx.Int32,
        i32_sk_eps: fx.Int32,
        i32_post_mult: fx.Int32,
        i32_norm_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m, i32_rms_eps, i32_pre_eps, i32_sk_eps, i32_post_mult, i32_norm_eps, arg_gp, arg_x1, arg_scale,
            arg_base, arg_nw, arg_post, arg_comb, arg_npre, arg_out, arg_codes, arg_qs,
        ).launch(grid=(fx.Int64(i32_m), 1, 1), block=(NTHR, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
