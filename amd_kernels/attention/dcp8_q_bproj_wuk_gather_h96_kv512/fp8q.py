# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Device helpers of the fp8 hi/lo query record (the layout of ``quantize_q`` in
attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048/reference.py): per (row, head) and block of 128 latent dims an E8M0
byte ``s = max(exponent(amax) - 7, 1)``, ``x = q * 2^(127 - s)``, ``hi = e4m3(x)`` and ``lo = e4m3((x - hi) * 32)``."""

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as rocdl_d
from flydsl.expr import rocdl
from flydsl.expr.typing import T

RECORD = 1168
Q_LO, Q_ROPE, Q_SCALE = 512, 1024, 1152


def _swap_pair(fn, x):
    st = llvm.StructType.get_literal([T.i32, T.i32])
    v = fx.Float32(x).bitcast(fx.Int32).ir_value()
    r = fn(st, v, v, False, False)
    a = fx.Int32(llvm.ExtractValueOp(T.i32, r, [0]).res).bitcast(fx.Float32)
    b = fx.Int32(llvm.ExtractValueOp(T.i32, r, [1]).res).bitcast(fx.Float32)
    return fx.Float32(a), fx.Float32(b)


def fmax(a, b):
    return fx.maxnumf(a, b, fastmath="nnan")


def xmax(x):
    """Max over the 4 lane groups of a wave (lanes l, l + 16, l + 32, l + 48)."""
    a, b = _swap_pair(rocdl.permlane16_swap, x)
    a, b = _swap_pair(rocdl.permlane32_swap, fmax(a, b))
    return fmax(a, b)


def e8m0_for(amax):
    """E8M0 byte s with amax / 2^(s - 127) in [128, 256), and f32 2^(127 - s)."""
    e = (fx.Float32(amax).bitcast(fx.Int32) >> 23) & 0xFF
    s = e - 7
    s = (s < 1).select(fx.Int32(1), s)
    inv = ((254 - s) << 23).bitcast(fx.Float32)
    return s, fx.Float32(inv)


def pack_fp8x4(vals):
    """4 f32 -> one i32 of e4m3 (byte i = vals[i])."""
    w = rocdl.cvt_pk_fp8_f32(
        T.i32, fx.Float32(vals[0]).ir_value(), fx.Float32(vals[1]).ir_value(), llvm.mlir_poison(T.i32), False
    )
    w = rocdl.cvt_pk_fp8_f32(T.i32, fx.Float32(vals[2]).ir_value(), fx.Float32(vals[3]).ir_value(), w, True)
    return fx.Int32(w)


def unpack_fp8x4(word):
    wd = fx.Int32(word).ir_value()
    a = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), wd, False))
    b = fx.Vector(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), wd, True))
    return [fx.Float32(a[0]), fx.Float32(a[1]), fx.Float32(b[0]), fx.Float32(b[1])]


def quant(vals, inv):
    """Vals * inv -> fp8 words (hi) and the residual * 32 as fp8 words (lo)."""
    x = [fx.Float32(v) * inv for v in vals]
    hi = [pack_fp8x4(x[4 * i : 4 * i + 4]) for i in range(len(x) // 4)]
    lo = []
    for i in range(len(hi)):
        dec = unpack_fp8x4(hi[i])
        lo.append(pack_fp8x4([(x[4 * i + j] - dec[j]) * 32.0 for j in range(4)]))
    return hi, lo


def bf16_pair(word):
    """A dword of two bf16 -> two f32 (low half first), exact."""
    w = fx.Int32(word)
    return fx.Float32((w << 16).bitcast(fx.Float32)), fx.Float32((w & fx.Int32(-65536)).bitcast(fx.Float32))


def abs_max_bits(words):
    """Largest |bf16| over a list of dwords of two bf16, as an f32."""
    m = fx.Int32(0)
    for w in words:
        w = fx.Int32(w)
        for v in ((w << 16) & 0x7FFFFFFF, w & 0x7FFF0000):
            m = (v > m).select(v, m)
    return fx.Float32(m.bitcast(fx.Float32))


def store_byte(val, rsrc, byte_off):
    v = std_arith.TruncIOp(ir.IntegerType.get_signless(8), fx.arith.unwrap(val)).result
    off = fx.arith.unwrap(fx.Int32(byte_off))
    rocdl_d.RawPtrBufferStoreOp(v, rsrc, off, fx.arith.constant(0), aux=ir.IntegerAttr.get(T.i32, 0))
