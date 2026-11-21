# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 packing, scalar helpers and the system-scope flag handshake of the eight-rank latent tail.

Adapted from norm/allreduce_attnres_rmsnorm_h7168_ranks8/norm_math.py of this repository.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

K = 3584  # latent width (the up projection's reduction)
N = 896  # this rank's hidden shard (7168 / 8)
WORLD = 8
SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores
NT = 0x2  # non-temporal

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def bf16x8(v):
    """8 fp32 of 4 dwords of bf16 (low half first)."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


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
    return fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(x)))


def sel8(p, a):
    """a[p] for a runtime p in [0, 8) over a list of eight values."""
    out = a[7]
    for i in range(6, -1, -1):
        out = (p == i).select(a[i], out)
    return out


def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def flag_store(addr, v):
    """An atomic store at system scope (a posted write-through: no read-modify-write at the remote memory)."""
    _llvm.StoreOp(
        fx.Int32(v).ir_value(),
        _gptr(addr).llvm_ptr,
        ordering=_llvm.AtomicOrdering.monotonic,
        syncscope="",
        alignment=4,
    )


def atomic_add(addr, v):
    """An atomic add at system scope, returning the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(v).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def flag_load(addr):
    """An atomic load at system scope (a spin on it is never hoisted)."""
    return fx.Int32(
        _llvm.LoadOp(
            T.i32,
            _gptr(addr).llvm_ptr,
            ordering=_llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


@flyc.jit
def spin_load(a, e):
    f = flag_load(a)
    while (f - e) < 0:
        f = flag_load(a)


@flyc.jit
def spin(a, e):
    f = atomic_add(a, 0)
    while (f - e) < 0:  # signed difference: the counts may wrap
        f = atomic_add(a, 0)


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
