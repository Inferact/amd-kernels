# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MFMA, LDS, FP4 convert and activation helpers shared by the expert GEMMs."""
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T

H, I, E, TOPK = 3584, 384, 896, 16  # noqa: E741
LOG2E = 1.4426950408889634
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0


def raw(x):
    return x.ir_value() if hasattr(x, "ir_value") else x


def mfma(a, b, acc, sa, sb, b_byte, transpose=False):
    """gfx950 FP4 x FP4 scaled 16x16x128 MFMA into AGPRs; ``transpose`` swaps the operands (C^T)."""
    ty = T.vec(4, T.f32)
    a_byte = 0
    if transpose:
        a, b, sa, sb = b, a, sb, sa
        a_byte, b_byte = b_byte, 0
    asm = (f"v_mfma_scale_f32_16x16x128_f8f6f4 $0, $1, $2, {'$0' if acc is not None else '0'}, $3, $4 "
           f"op_sel:[{a_byte % 2},{b_byte % 2},0] op_sel_hi:[{a_byte // 2},{b_byte // 2},0] cbsz:4 blgp:4")
    ops = [raw(a), raw(b), raw(sa), raw(sb)]
    cons = "=a,v,v,v,v"
    if acc is not None:
        ops.append(raw(acc))
        cons += ",0"
    return llvm.inline_asm(ty, ops, asm, cons, has_side_effects=True)


def hold(xs):
    """Wait out the MFMA latency before any accumulator is read."""
    ty = ir.Type.parse("!llvm.struct<(" + ", ".join(["vector<4xf32>"] * len(xs)) + ")>")
    cons = ",".join(["=a"] * len(xs) + [str(i) for i in range(len(xs))])
    v = llvm.inline_asm(ty, [raw(x) for x in xs], "s_nop 7\ns_nop 7\ns_nop 7", cons, has_side_effects=True)
    return [llvm.extractvalue(T.vec(4, T.f32), v, [i]) for i in range(len(xs))]


def lds_ptr(base, off):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), raw(fx.Int64(base + off)))


def cvt_fp4(old, a, b, scale, sel):
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.cvt.scalef32.pk.fp4.f32",
                                        [raw(old), raw(a), raw(b), raw(scale), raw(fx.Int32(sel))], [], []))


def e8m0_roundup(amax):
    """aiter's e8m0 rule: ceil exponent of amax / 6, saturating at 254."""
    wi = (amax * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
    bexp = (wi + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23)) & fx.Int32(0xFF)
    return (fx.Uint32(bexp) < 254).select(bexp, fx.Int32(254))


def sigmoid_f32(x):
    e = fx.exp2(x * fx.Float32(-LOG2E), fastmath="afn")
    return fx.Float32(fx.rocdl.rcp(T.f32, raw(fx.Float32(1.0) + e)))


def tanh_f32(x):
    e = fx.exp2(fx.max(x, -x) * fx.Float32(-2.0 * LOG2E), fastmath="afn")
    r = fx.Float32(fx.rocdl.rcp(T.f32, raw(fx.Float32(1.0) + e)))
    t = (fx.Float32(1.0) - e) * r
    return (x > fx.Float32(0.0)).select(t, -t)


def situ(g, u):
    """SiTUv2, unclamped: beta tanh(g / beta) sigmoid(g) * (linear_beta tanh(u / linear_beta))."""
    gate = fx.Float32(SITU_BETA) * tanh_f32(g * fx.Float32(1.0 / SITU_BETA)) * sigmoid_f32(g)
    up = fx.Float32(SITU_LINEAR_BETA) * tanh_f32(u * fx.Float32(1.0 / SITU_LINEAR_BETA))
    return gate * up
