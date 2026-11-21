# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Building blocks of the sparse MLA decode kernels: LDS and global access, MFMA wrappers, fp8 quantization of q, lane
reductions.

Adapted from ROCm/aiter, aiter/ops/flydsl/kernels/sparse_mla_fp8ds_gfx950.py @ a61e02c26a3915011e85b103574c25d457890825.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, ptr_buf_tensor
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
RECORD_BYTES = KV_LORA_RANK + 16 + 2 * QK_ROPE_HEAD_DIM  # 656
HEADS = 96
HG_HEADS = 16
N_HG = HEADS // HG_HEADS
XCDS = 8
LN2 = 0.6931471805599453
# 2^7 as f32 bits: P' = P * scale <= 2^15 stays under the fp16 max.
SCALE_CAP_BITS = 0x43000000
_L = KV_LORA_RANK


def _lds_ptr(elem, addr, align):
    return fx.inttoptr(
        fx.PointerType.get(elem, fx.AddressSpace.Shared, align), fx.Int32(addr)
    )


def _lds_load(elem, n, addr, align):
    return Vec(fx.ptr_load(_lds_ptr(elem, addr, align), result_type=T.vec(n, elem)))


def _lds_store(vec, elem, addr, align):
    fx.ptr_store(Vec(vec).ir_value(), _lds_ptr(elem, addr, align))


def _lds_llvm_ptr(addr):
    return fx.to_llvm_ptr(
        fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(addr))
    )


def _global_llvm_ptr(addr):
    return fx.to_llvm_ptr(
        fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Global), fx.Int64(addr))
    )


def _global_i32(addr, words):
    """``words`` i32 (1 or 4) at the 64-bit global address ``addr``: a cache
    past 4 GiB is out of a buffer descriptor's 32-bit offset reach."""
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4 * words), fx.Int64(addr)
    )
    if words == 1:
        return fx.Int32(fx.generic_load(ptr, dtype=fx.Int32).ir_value())
    return Vec(fx.generic_load(ptr, dtype=fx.Vector[fx.Int32, words]).ir_value())


def _tr8(addr, imm):
    ptr = _lds_llvm_ptr(fx.Int32(addr) + imm)
    return Vec(rocdl.ds_read_tr8_b64_(T.vec(2, T.i32), ptr))


def _cat(a, b, n):
    return Vec(a).shuffle(Vec(b), list(range(2 * n)))


def _mfma_fp8(a, b, c, sa, sb):
    return Vec(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).ir_value()),
                _raw(Vec(b).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
                _raw(fx.Int32(sa)),
                0,
                _raw(fx.Int32(sb)),
            ],
        )
    )


def _mfma_f16(a, b, c):
    return Vec(
        rocdl.mfma_f32_16x16x32_f16(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).bitcast(fx.Float16).ir_value()),
                _raw(Vec(b).bitcast(fx.Float16).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
            ],
        )
    )


def _mfma_bf16(a, b, c):
    return Vec(
        rocdl.mfma_f32_16x16x32_bf16(
            T.vec(4, T.f32),
            [
                _raw(Vec(a).bitcast(fx.BFloat16).ir_value()),
                _raw(Vec(b).bitcast(fx.BFloat16).ir_value()),
                _raw(Vec(c).ir_value()),
                0,
                0,
                0,
            ],
        )
    )


def _cvt_v8(x):
    """8 raw e4m3 (2 i32 from tr8) -> 8 f16 (4 i32), exact."""
    one = _raw(fx.Float32(1.0).ir_value())
    out = []
    for i in range(2):
        wd = _raw(fx.Int32(Vec(x)[i]).ir_value())
        for hs in (False, True):
            out.append(Vec(rocdl.cvt_scalef32_pk_f16_fp8(T.vec(2, T.f16), wd, one, hs)))
    return _cat(_cat(out[0], out[1], 2), _cat(out[2], out[3], 2), 4).bitcast(fx.Int32)


def _pack_fp8x4(vals):
    """4 f32 -> one i32 of e4m3 (byte i = vals[i])."""
    w = rocdl.cvt_pk_fp8_f32(
        T.i32,
        _raw(fx.Float32(vals[0]).ir_value()),
        _raw(fx.Float32(vals[1]).ir_value()),
        llvm.mlir_poison(T.i32),
        False,
    )
    w = rocdl.cvt_pk_fp8_f32(
        T.i32,
        _raw(fx.Float32(vals[2]).ir_value()),
        _raw(fx.Float32(vals[3]).ir_value()),
        w,
        True,
    )
    return fx.Int32(w)


def _unpack_fp8x4(word):
    wd = _raw(fx.Int32(word).ir_value())
    a = Vec(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), wd, False))
    b = Vec(rocdl.cvt_pk_f32_fp8(T.vec(2, T.f32), wd, True))
    return [fx.Float32(a[0]), fx.Float32(a[1]), fx.Float32(b[0]), fx.Float32(b[1])]


def _quant(vals, inv):
    """Vals * inv -> fp8 words (hi), and the residual * 32 as fp8 words (lo,
    whose E8M0 scale is the hi one minus 5)."""
    x = [fx.Float32(v) * inv for v in vals]
    hi = [_pack_fp8x4(x[4 * i : 4 * i + 4]) for i in range(len(x) // 4)]
    lo = []
    for i in range(len(hi)):
        dec = _unpack_fp8x4(hi[i])
        lo.append(_pack_fp8x4([(x[4 * i + j] - dec[j]) * 32.0 for j in range(4)]))
    return hi, lo


def _lo_scale(s):
    s = fx.Int32(s) - 5
    return (s < 0).select(fx.Int32(0), s)


def _swap_pair(fn, x):
    st = llvm.StructType.get_literal([T.i32, T.i32])
    v = _raw(fx.Float32(x).bitcast(fx.Int32).ir_value())
    r = fn(st, v, v, False, False)
    a = fx.Int32(llvm.ExtractValueOp(T.i32, r, [0]).res).bitcast(fx.Float32)
    b = fx.Int32(llvm.ExtractValueOp(T.i32, r, [1]).res).bitcast(fx.Float32)
    return fx.Float32(a), fx.Float32(b)


def _fmax(a, b):
    return fx.maxnumf(a, b, fastmath="nnan")


def _xsum(x):
    """Sum over the 4 lane groups (xor 16, 32) via permlane swaps."""
    a, b = _swap_pair(rocdl.permlane16_swap, x)
    a, b = _swap_pair(rocdl.permlane32_swap, a + b)
    return a + b


def _xmax(x):
    """Max over the 4 lane groups (xor 16, 32) via permlane swaps."""
    a, b = _swap_pair(rocdl.permlane16_swap, x)
    a, b = _swap_pair(rocdl.permlane32_swap, _fmax(a, b))
    return _fmax(a, b)


def _imax(a, b):
    return (a > b).select(a, b)


def _imin(a, b):
    return (a < b).select(a, b)


def _readfirstlane(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


def _e8m0_for(amax):
    """E8M0 byte s with amax / 2^(s - 127) in [128, 256), and f32 2^(127 - s)."""
    e = (fx.Float32(amax).bitcast(fx.Int32) >> 23) & 0xFF
    s = e - 7
    s = (s < 1).select(fx.Int32(1), s)
    inv = ((254 - s) << 23).bitcast(fx.Float32)
    return s, fx.Float32(inv)


def _q_load(q_t, q_row, g):
    """Issue this lane's q loads: raw bf16 vectors, two lists (record blocks g // 2 and 2 + g // 2)."""
    raw = ([], [])
    for j in range(4):
        c = 2 * j + g % 2
        for hb, dst in ((0, raw[0]), (2, raw[1])):
            d0 = (hb + g // 2) * 128 + c * 16  # bf16 dims
            for k in range(2):
                dst.append(Vec(q_t.load(q_row + d0 // 2 + k * 4, vec_size=4)))
    return raw


def _q_prologue(raw, g):
    """Quantize this lane's q row to fp8 hi/lo MFMA operands.

    MFMA scale group kb (lanes 16kb.. supply its E8M0) is K bytes
    32kb..32kb + 31, so lane d's bytes 0-15 belong to group d // 2 and bytes
    16-31 to group 2 + d // 2. Lane g therefore holds, for MFMA j, chunk
    2j + g % 2 of record block g // 2 (first 16 B) and of block 2 + g // 2
    (second 16 B), and supplies the scale of block g.

    Returns (q8 hi operands, hi E8M0, q8 lo operands, lo E8M0).
    """
    va = [[fx.Float32(x) for x in Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)] for v in raw[0]]
    vb = [[fx.Float32(x) for x in Vec(v.bitcast(fx.BFloat16)).to(fx.Float32)] for v in raw[1]]
    va = [x for blk in va for x in blk]
    vb = [x for blk in vb for x in blk]
    am = []
    for vv in (va, vb):
        a = fx.Float32(0.0)
        for x in vv:
            a = _fmax(a, _fmax(x, fx.Float32(0.0) - x))
        p0, p1 = _swap_pair(rocdl.permlane16_swap, a)
        am.append(_fmax(p0, p1))  # amax of block g // 2 (va) / 2 + g // 2 (vb)
    oth = []
    for a in am:
        p0, p1 = _swap_pair(rocdl.permlane32_swap, a)
        oth.append((g >= 2).select(p0, p1))  # value of lane g ^ 2
    lo_half = g < 2
    blk = [
        lo_half.select(am[0], oth[0]),
        lo_half.select(oth[0], am[0]),
        lo_half.select(am[1], oth[1]),
        lo_half.select(oth[1], am[1]),
    ]
    mine = (g == 0).select(
        blk[0], (g == 1).select(blk[1], (g == 2).select(blk[2], blk[3]))
    )
    _, inva = _e8m0_for(am[0])
    _, invb = _e8m0_for(am[1])
    q8, q8l = [], []
    for j in range(4):
        ha, la = _quant(va[16 * j : 16 * j + 16], inva)
        hb, lb = _quant(vb[16 * j : 16 * j + 16], invb)
        q8.append(Vec.from_elements(ha + hb, fx.Int32))
        q8l.append(Vec.from_elements(la + lb, fx.Int32))
    qs, _ = _e8m0_for(mine)
    return q8, qs, q8l, _lo_scale(qs)


class _BoundedI32(GTensor):
    """GTensor whose buffer descriptor carries the real byte size, so an
    out-of-range read returns 0 instead of touching memory past the end."""

    def __init__(self, memref, nbytes):
        super().__init__(memref, dtype=T.i32, shape=(-1,))
        self.nbytes = nbytes

    def _view(self, elem, unit_elems):
        return ptr_buf_tensor(
            self.base_i64,
            elem,
            unit_elems=unit_elems,
            unit_stride=1,
            num_records_bytes=fx.Int64(self.nbytes),
        )


def _guarded(pred, fn):
    """Run fn() under a runtime predicate (an scf.if around its body)."""

    @flyc.jit
    def _do(_pred=pred, _fn=fn):
        if _pred:
            _fn()

    _do()
