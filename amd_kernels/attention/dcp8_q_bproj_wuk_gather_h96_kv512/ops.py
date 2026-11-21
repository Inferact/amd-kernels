# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shared device helpers."""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec


def mfma(a, b, c):
    return Vec(
        fx.rocdl.mfma_f32_16x16x32_bf16(
            Vec.make_type(4, fx.Float32), [a.bitcast(fx.BFloat16), b.bitcast(fx.BFloat16), c, 0, 0, 0]
        )
    )


def pk_bf16(a, b):
    """Two fp32 to a dword of two bf16 (RNE, ``a`` low) by a native fptrunc: unlike ``rocdl.cvt_pk_bf16_f32`` (inline
    asm) LLVM inserts the wait states after the MFMA that wrote ``a`` and ``b``."""
    return fx.Int32(Vec.from_elements([fx.Float32(a), fx.Float32(b)], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0])


def now():
    """s_memrealtime (100 MHz) as an Int64."""
    return fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def mfma_stub(a, b, c):
    """The MFMA's operand dependence without its math (the skeleton kernels that bound the measured latency chain)."""
    return Vec(Vec(c).bitcast(fx.Int32) ^ Vec(a) ^ Vec(b)).bitcast(fx.Float32)
