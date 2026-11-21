# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 packing, the 16x16x32 bf16 MFMA wrapper and LDS pointers shared by the scan kernel."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def sched_barrier():
    fx.rocdl.sched_barrier(0)


def barrier():
    llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)


def lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), raw(fx.Int64(byte_addr)))


def lds_load4(byte_addr):
    """16 bytes of LDS as four i32."""
    return Vec(llvm.load(T.vec(4, T.i32), lds_ptr(byte_addr), alignment=16))


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (RNE), ``a`` in the low half."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def bf16x2_lo(w):
    return (fx.Int32(w) << 16).bitcast(fx.Float32)


def bf16x2_hi(w):
    return (fx.Int32(w) & -65536).bitcast(fx.Float32)


def mfma(a, b, acc):
    """v_mfma_f32_16x16x32_bf16: ``a`` / ``b`` are 8 bf16 (4 dwords) per lane, D[4q + e][i] for lane (i, q)."""
    a = Vec(a).bitcast(fx.BFloat16)
    b = Vec(b).bitcast(fx.BFloat16)
    return Vec(fx.rocdl.mfma_f32_16x16x32_bf16(Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0]))


def lds_load2(byte_addr):
    """8 bytes of LDS as two i32."""
    return Vec(llvm.load(T.vec(2, T.i32), lds_ptr(byte_addr), alignment=8))


def lds_store_i16(byte_addr, value):
    """Store the low 16 bits of an i32 to LDS."""
    llvm.store(raw(fx.Int32(value).to(fx.Int16)), lds_ptr(byte_addr), alignment=2)


def touch_args(*args):
    """Read every kernel argument at the top: one batch of scalar loads instead of one dependent round per use."""
    vals = [raw(a) for a in args]
    llvm.inline_asm(None, vals, "", ",".join("s" for _ in vals), has_side_effects=True)


FLAG_STRIDE = 4  # int32 per hand-over flag
WALK_GROUPS = 4  # 4 or 8: walk workgroups (value slice groups) per (sequence, head)
