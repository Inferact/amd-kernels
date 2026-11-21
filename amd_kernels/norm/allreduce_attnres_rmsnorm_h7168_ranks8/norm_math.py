# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 packing, wave reductions, LDS and peer-memory primitives of the eight-rank kernel.

The DPP wave sum and the bf16 pack are adapted from norm/allreduce_gemma_bf16_mxfp8_h6144/norm_math.py of this repository.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.typing import Vector as Vec

H = 7168
WORLD = 8
SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b)))
    )


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _dpp_f(v, ctrl, row_mask=0xF, old=None):
    """v_mov_b32_dpp of an fp32; rows ``row_mask`` leaves out keep ``old``, out-of-row sources read 0."""
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(_bits(old))
    r = _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(_bits(v)), ctrl, row_mask, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def wave_sum_lane63(acc):
    """Sum of the 64 lanes, valid in lane 63: row_shr 8/4/2/1 adds, row_bcast 15 and 31."""
    for sh in (8, 4, 2, 1):
        acc = acc + _dpp_f(acc, 0x110 + sh)
    acc = _dpp_f(acc, 0x142, row_mask=0xA, old=acc) + acc
    return acc + _dpp_f(acc, 0x143)


def rsq(x):
    return fx.Float32(_rocdl_ops.rsq(T.f32, as_ir_value(x)))


def exp2(x):
    return fx.Float32(_rocdl_ops.exp2(T.f32, as_ir_value(x)))


def sel8(p, a):
    """a[p] for a runtime p in [0, 8) over a list of eight values."""
    out = a[7]
    for i in range(6, -1, -1):
        out = (p == i).select(a[i], out)
    return out


def load_ptrs(addr):
    """The 8 int64 at ``addr`` (scalar loads)."""
    tr = buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=WORLD * 8)
    out = []
    for h in range_constexpr(4):
        v = Vec(buffer_ops.buffer_load(tr, h * 4, vec_width=4, is_scalar=True))
        for i in range_constexpr(2):
            lo = fx.Int64(fx.Int32(v[2 * i])) & fx.Int64(0xFFFFFFFF)
            out.append(lo | (fx.Int64(fx.Int32(v[2 * i + 1])) << 32))
    return out


def _global_ptr(addr):
    return _llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))).result


def vstore(addr, v):
    """A non-temporal 16 B global store of 4 i32 at byte address ``addr`` (the mailboxes are uncached memory)."""
    _llvm.StoreOp(as_ir_value(v), _global_ptr(addr), nontemporal=True, alignment=16)


def compiler_barrier():
    """Memory clobber: the loads of a poll loop are not hoisted out of it."""
    _llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)


def lds_ptr(addr):
    return _llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), as_ir_value(fx.Int64(addr)))


def lds_store16(addr, v):
    _llvm.StoreOp(as_ir_value(v), lds_ptr(addr), alignment=16)


def lds_load16(addr):
    return Vec(_llvm.load(T.vec(4, T.i32), lds_ptr(addr), alignment=16))


def lds_arrive(addr):
    """Add 1 to the LDS counter at ``addr`` (release: this wave's earlier LDS writes are visible first)."""
    _llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, lds_ptr(addr), fx.Int32(1).ir_value(), _llvm.AtomicOrdering.release,
                      syncscope="workgroup", alignment=4)


def _lds_read_counter(addr):
    return fx.Int32(_llvm.AtomicRMWOp(_llvm.AtomicBinOp.add, lds_ptr(addr), fx.Int32(0).ir_value(),
                                      _llvm.AtomicOrdering.acquire, syncscope="workgroup", alignment=4).result)


@flyc.jit
def lds_wait(addr, target):
    """Spin until the LDS counter at ``addr`` reaches ``target`` (acquire: later LDS reads stay after it)."""
    f = _lds_read_counter(addr)
    while f < fx.Int32(target):
        f = _lds_read_counter(addr)
