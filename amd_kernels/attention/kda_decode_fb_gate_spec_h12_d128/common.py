# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Packed bf16 helpers, DPP reductions and the fast transcendentals of the KDA decode kernel."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr.typing import T

LOG2E = 1.4426950408889634


def bf16x2_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf16x2_hi(w):
    return (fx.Int32(w) & -65536).bitcast(fx.Float32)


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (RNE), ``a`` in the low half."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def round_bf16(x):
    """fp32 -> bf16 -> fp32 (RNE, finite inputs); integer ops, not ``pk_bf16``: an MFMA result feeds it, and inline asm
    is invisible to the hazard recognizer."""
    bits = fx.Float32(x).bitcast(fx.Int32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & -65536).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def dpp(v, ctrl, row_mask=0xF, old=None):
    """v_mov_b32_dpp of an fp32; source lanes outside the row read 0, rows ``row_mask`` leaves out keep ``old``."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(_bits(old))
    r = _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, row_mask, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def sum_group(v, n):
    """Sum of ``v`` over the ``n`` (1, 2, 4, 8 or 16) lanes of an aligned group, in every lane."""
    for ctrl in (0xB1, 0x4E, 0x141, 0x140)[: n.bit_length() - 1]:
        v = v + dpp(v, ctrl)
    return v


def wave_sum_lane63(acc):
    """Sum of ``acc`` over the 64 lanes, valid in lane 63 (row_shr 8/4/2/1 adds, row_bcast 15 and 31)."""
    for sh in (8, 4, 2, 1):
        acc = acc + dpp(acc, 0x110 + sh)
    acc = dpp(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + dpp(acc, 0x143)


def exp2(x):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.as_ir_value(fx.Float32(x))))


def rcp(x):
    return fx.Float32(fx.rocdl.rcp(T.f32, fx.as_ir_value(fx.Float32(x))))


def sigmoid(x):
    return rcp(fx.Float32(1.0) + exp2(x * fx.Float32(-LOG2E)))


def rsqrt(x):
    return fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(fx.Float32(x))))


def fma(a, b, c):
    return fx.math.fma(a, b, c)


def mfma_bf16(a4, b4, acc):
    """v_mfma_f32_16x16x32_bf16: lane (i, q) gives A[i][8q:8q+8], B[8q:8q+8][i] (4 dwords each), gets D[4q + e][i]."""
    a = fx.Vector(a4).bitcast(fx.BFloat16)
    b = fx.Vector(b4).bitcast(fx.BFloat16)
    return fx.Vector(fx.rocdl.mfma_f32_16x16x32_bf16(fx.Vector.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]))


def global_i32_ptr(addr_i64):
    """Typed global i32 pointer at a raw device address."""
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def global_atomic_add(addr_i64, value):
    """``old = *addr; *addr += value`` on a global i32 (agent scope, relaxed)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(value).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )
