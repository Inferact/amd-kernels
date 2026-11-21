# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""System-scope helpers of the push epilogue: the cache modifier of stores that peers read and the atomic ticket / flag
words (uncached memory); tickets and the sequence counter are counters, never output adds."""
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

SYS_CM = 0x11  # sc0 sc1: system scope


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def flag_add(addr_i64, v):
    """Atomic add at system scope on an uncached word; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _global_i32_ptr(addr_i64).llvm_ptr,
            fx.Int32(v).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )
