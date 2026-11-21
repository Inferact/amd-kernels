# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 unpacking, DPP folds and the power-of-two group scale shared by both kernels."""

import functools

import flydsl.expr as fx
from flydsl._mlir import ir as _ir
import torch
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr.typing import T

PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}


def f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def bf16x8(v):
    """8 fp32 of 4 dwords of bf16 (low half first)."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def pk_bf16(a, b):
    """Two fp32 -> two bf16 in a dword (v_cvt_pk_bf16_f32, RNE), ``a`` low."""
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )


def maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def tree_absmax(vals):
    """max |v| of a list of fp32, as a balanced tree (a short dependent chain)."""
    a = [fx.math.absf(v) for v in vals]
    while len(a) > 1:
        a = [maxnum(a[i], a[i + 1]) if i + 1 < len(a) else a[i] for i in range(0, len(a), 2)]
    return a[0]


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(
        _rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(src), ctrl, 0xF, bank_mask, full)
    )


def xor_lane(v, m):
    """Lane ``l ^ m`` (m = 1, 2, 4, 8) of an i32."""
    if m == 1:
        return _dpp(v, 0xB1)
    if m == 2:
        return _dpp(v, 0x4E)
    if m == 4:
        return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)
    assert m == 8
    return _dpp(v, 0x128)


def row_max(x):
    """Max over the 16 lanes of a DPP row, in every lane."""
    for m in (1, 2, 4, 8):
        x = maxnum(x, xor_lane(bits(x), m).bitcast(fx.Float32))
    return x


def _swap(v, op):
    pair_ty = _ir.Type.parse("!llvm.struct<(i32, i32)>")
    res = op(pair_ty, fx.as_ir_value(v), fx.as_ir_value(v), False, False)
    return fx.Int32(_llvm.extractvalue(T.i32, res, [0])), fx.Int32(_llvm.extractvalue(T.i32, res, [1]))


def wave_allsum(x):
    """Sum over the 64 lanes of a wave in every lane, without LDS or SGPR hops (DPP xor 1..8, permlane16/32 swaps).
    Every lane adds the same operands in the same order, so all lanes hold the same bits."""
    for m in (1, 2, 4, 8):
        x = x + xor_lane(bits(x), m).bitcast(fx.Float32)
    for op in (_rocdl_ops.permlane16_swap, _rocdl_ops.permlane32_swap):
        a, b = _swap(bits(x), op)
        x = a.bitcast(fx.Float32) + b.bitcast(fx.Float32)
    return x


def wave_sum(x):
    """Sum over the 64 lanes of a wave, in every lane (row_shr folds, then lane 63's total)."""
    for sh in (8, 4, 2, 1):
        x = x + _dpp(bits(x), 0x110 + sh).bitcast(fx.Float32)
    r = [
        fx.Int32(fx.rocdl.readlane(T.i32, bits(x), fx.Int32(16 * i + 15))).bitcast(fx.Float32)
        for i in range(4)
    ]
    return (r[0] + r[1]) + (r[2] + r[3])


def pow2_exponent(s):
    """Biased exponent of the smallest power of two >= the positive normal fp32 ``s``."""
    b = bits(s)
    exact = (b & 0x7FFFFF) == 0
    return ((b >> 23) & 255) + exact.select(fx.Int32(0), fx.Int32(1))


def pow2_pair(e):
    """(2^(e - 127), 2^(127 - e)) as fp32 for a biased exponent e in [1, 253]."""
    return f32(e << 23), f32((254 - e) << 23)


@functools.cache
def on_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
