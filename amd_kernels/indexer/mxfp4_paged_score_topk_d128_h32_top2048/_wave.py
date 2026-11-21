# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# The permlane / DPP wave primitives follow AITER's FlyDSL FP4 MQA kernels
# (aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_rowgroup.py, MIT, Copyright (C) 2024-2026, Advanced Micro
# Devices, Inc.).
"""Wave-level FlyDSL primitives of the scorer and the selection: scans, ballots, the order-preserving score key."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32

I, F, I64 = fx.Int32, fx.Float32, fx.Int64
SC1 = 16  # cache modifier: agent-scope (write-through stores, coherent loads); the XCDs' L2s are not coherent
AtomicBinOp = llvm.AtomicBinOp


def atomic_agent(op, addr_i64, value):
    """``old = *addr; *addr (op)= value`` on a global i32, device (agent) scope, relaxed."""
    ptr = fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4), I64(addr_i64))
    return I(llvm.AtomicRMWOp(op, ptr.llvm_ptr, I(value).ir_value(), llvm.AtomicOrdering.monotonic, syncscope="agent",
                              alignment=4).result)
INT_MIN = -(1 << 31)
INT_MAX = (1 << 31) - 1


def permlane_swap(swap, x, y):
    """v_permlane{32,16}_swap on f32 registers -> (x', y')."""
    pair = swap(llvm.StructType.get_literal([T.i32, T.i32]), as_ir_value(x.bitcast(I)), as_ir_value(y.bitcast(I)), False, False)
    return tuple(I(llvm.extractvalue(T.i32, pair, [j])).bitcast(F) for j in range(2))


def vop(asm, *args):
    """One unpacked f32 VALU op in asm (the SLP vectorizer cannot pair it into a packed op)."""
    return F(llvm.inline_asm(T.f32, [F(a).ir_value() for a in args], asm, "=v" + ",v" * len(args), has_side_effects=False))


def global_i32(addr, words, nontemporal=False):
    """``words`` i32 at the 64-bit global address ``addr`` (a pool past 4 GiB is reached only this way)."""
    ptr = fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4 * words), I64(addr))
    return fx.generic_load(ptr, dtype=fx.Vector[I, words], nontemporal=nontemporal).ir_value()


def relu2(x):
    """relu(x), one VALU op (AITER's 2 relu(x) = x + |x| with the 1/2 in the head weights is the same product bit for bit)."""
    return fx.max(F(x), F(0.0))


def inclusive_scan(v):
    """Running sum over the lanes of a wave (row_shr 1, 2, 4, 8 within each 16-lane row, then the rows' totals)."""
    v = I(v)
    zero = I(0)
    for k in (1, 2, 4, 8):
        v = v + I(update_dpp_i32(zero, v, 0x110 + k, 0xF, 0xF, False))
    v = v + I(update_dpp_i32(zero, v, 0x142, 0xA, 0xF, False))
    return v + I(update_dpp_i32(zero, v, 0x143, 0xC, 0xF, False))


def lane_xor(v, off):
    """Lane ``lane ^ off``'s ``v`` (i32, off 1 - 8): DPP within a 16-lane row."""
    if off == 8:
        y = update_dpp_i32(v, v, 0x118, 0xF, 0xC, False)
        return I(update_dpp_i32(y, v, 0x108, 0xF, 0x3, False))
    if off == 4:
        y = update_dpp_i32(v, v, 0x114, 0xF, 0xA, False)
        return I(update_dpp_i32(y, v, 0x104, 0xF, 0x5, False))
    if off == 2:
        return I(update_dpp_i32(v, v, 0x4E, 0xF, 0xF, False))
    return I(update_dpp_i32(v, v, 0xB1, 0xF, 0xF, False))


def wave_reduce(v, op):
    """``op`` (``fx.max``, ``fx.min``) over the 64 lanes: permlane swaps across the rows, DPP within them."""
    v = I(v)
    for swap in (rocdl.permlane32_swap, rocdl.permlane16_swap):
        a, d = permlane_swap(swap, v.bitcast(F), v.bitcast(F))
        v = op(a.bitcast(I), d.bitcast(I))
    for off in (8, 4, 2, 1):
        v = op(lane_xor(v, off), v)
    return v


def readlane(v, lane):
    return I(rocdl.readlane(T.i32, I(v), I(lane)))


def uniform(v):
    return I(rocdl.readfirstlane(T.i32, I(v)))


def ballot(pred):
    return I64(rocdl.ballot(T.i64, pred))


def score_key(bits):
    """Order-preserving signed key of an fp32 (as int32 bits): larger float, larger key."""
    return (bits < 0).select(bits ^ I(0x7FFFFFFF), bits)


def key_to_float(key):
    return (key < 0).select(key ^ I(0x7FFFFFFF), key).bitcast(F)


class _View:
    def __init__(self, ptr):
        self.ptr = ptr


def layout(**sizes):
    """Word offsets (16-byte aligned) of fields of the given sizes in one block, and the block's size."""
    offs, at = {}, 0
    for name, n in sizes.items():
        offs[name] = at
        at += -(-n // 4) * 4
    return offs, at


def carve(raw, **fields):
    """Views (``.ptr``) of fields at word offsets of one shared array: roles of a fused kernel share one LDS block."""
    from types import SimpleNamespace

    return SimpleNamespace(**{name: _View(fx.add_offset(raw.ptr, off)) for name, off in fields.items()})


def store_lds(field, idx, v):
    """``field.ptr[idx] = v`` in a function, so a store under a dynamic ``if`` does not rebind the shared handle."""
    field.ptr[idx] = v


def now():
    """s_memrealtime (100 MHz)."""
    return I64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def popcount(mask):
    return I(fx.math.ctpop(mask).to(I))


def lane_prefix(mask):
    """Set bits of the 64-bit ``mask`` below this lane."""
    lo = mask.to(I)
    hi = (mask >> I64(32)).to(I)
    return I(rocdl.mbcnt_hi(T.i32, as_ir_value(hi), as_ir_value(I(rocdl.mbcnt_lo(T.i32, as_ir_value(lo), as_ir_value(I(0)))))))
