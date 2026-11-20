# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact BF16 rows of 6144 values in 9600 bytes (0.781x) for the link: the two-rank decode all-reduce's 12-bit code.

Adapted from amd_kernels norm/gemma_add_rmsnorm_h6144 (allreduce_add_rmsnorm/lossless_bf16.py, the codec, and
allreduce_add_rmsnorm_ranks2/lossless_x3.py, the record layout).

Code lane t holds columns 8 t .. 8 t + 7; a virtual wave is 64 code lanes (512 values). Per 32 values a group
exponent; per value the sign, 7 mantissa bits and a 4-bit exponent delta 0..14; delta 15 is a signed zero or an index
into the virtual wave's six exception values. A lane with several exceptions, or a virtual wave with more than six,
stores its 1 KB raw in the row's raw slot (a bf16 row: the values at their own offsets). Lane t's 8 sign/mantissa
bytes and 8 nibbles are a 12-byte record at byte 12 t, then 12 headers of 32 B at ``HDR`` (16 group exponent bytes,
the count word, 6 exception values). Every 16-bit pattern round-trips.

``dec16`` reads with the norm lane map (thread tx holds columns 16 tx .. 16 tx + 15, code lanes 2 tx and 2 tx + 1):
a hardware wave then spans two virtual waves, so the zero / exception / raw paths branch per lane.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .kernel import SYS

H = 6144
ROWB = H * 2  # a raw (bf16) row
CODED = 9600  # a coded row
HDR = 9216  # its headers
VWAVES = 12
NIBPOS = [4 * (v // 2) + 16 * (v % 2) for v in range(8)]
RAWF, SPECF, NEXC = 0x100, 0x200, 6
SIGN2 = -2147450880  # 0x80008000


def _dpp_i(v, ctrl):
    return fx.Int32(_rocdl_ops.update_dpp(
        T.i32, _llvm.mlir_undef(T.i32), fx.as_ir_value(fx.Int32(v)), ctrl, 0xF, 0xF, True
    ))


def _imax(a, b):
    return (a > b).select(a, b)


@flyc.jit
def enc(value, lane):
    """Code one lane's eight BF16 values; return the wave header and payload."""
    one, zero = fx.Int32(1), fx.Int32(0)
    rd = [fx.Int32(value[t]) for t in range_constexpr(4)]
    us = []
    for t in range_constexpr(4):
        us.append(rd[t] & 0xFFFF)
        us.append((rd[t] >> 16) & 0xFFFF)
    ex = [(u >> 7) & 0xFF for u in us]
    gmax = ex[0]
    for t in range_constexpr(1, 8):
        gmax = _imax(gmax, ex[t])
    gmax = _imax(gmax, _dpp_i(gmax, 0xB1))  # quad_perm [1, 0, 3, 2]
    gmax = _imax(gmax, _dpp_i(gmax, 0x4E))  # quad_perm [2, 3, 0, 1]
    dd = [gmax - ex[t] for t in range_constexpr(8)]
    big = [dd[t] > 14 for t in range_constexpr(8)]
    exc = [big[t] & ((us[t] & 0x7FFF) != 0) for t in range_constexpr(8)]
    nib = fx.Int32(0)
    nbig = fx.Int32(0)
    nexc = fx.Int32(0)
    ev = fx.Int32(0)
    for t in range_constexpr(8):
        nib = nib | (big[t].select(fx.Int32(15), dd[t]) << NIBPOS[t])
        nbig = nbig + big[t].select(one, zero)
        nexc = nexc + exc[t].select(one, zero)
        ev = ev | exc[t].select(us[t], zero)
    has = nexc > 0
    vsp = fx.Int64(fx.rocdl.ballot(T.i64, nbig > 0))
    v1 = fx.Int64(fx.rocdl.ballot(T.i64, has))
    av = [((us[t] >> 8) & 0x80) | (us[t] & 0x7F) for t in range_constexpr(8)]
    a0 = av[0] | (av[1] << 8) | (av[2] << 16) | (av[3] << 24)
    a1 = av[4] | (av[5] << 8) | (av[6] << 16) | (av[7] << 24)
    cnt = zero
    rawi = zero
    idx = one
    if v1 != fx.Int64(0):
        # an exception in this wave (wave-uniform, rare): list indices, or a raw wave
        v2 = fx.Int64(fx.rocdl.ballot(T.i64, nexc > 1))
        cnt = fx.Int64(fx.math.ctpop(v1)).to(fx.Int32)
        rawi = ((v2 != fx.Int64(0)) | (cnt > NEXC)).select(one, zero)
        below = v1 & ((fx.Int64(1) << fx.Int64(lane)) - fx.Int64(1))
        idx = fx.Int64(fx.math.ctpop(below)).to(fx.Int32) + 1
        ax = [exc[t].select(((us[t] >> 8) & 0x80) | idx, av[t]) for t in range_constexpr(8)]
        a0 = ax[0] | (ax[1] << 8) | (ax[2] << 16) | (ax[3] << 24)
        a1 = ax[4] | (ax[5] << 8) | (ax[6] << 16) | (ax[7] << 24)
    rawb = rawi != 0
    cwv = rawb.select(fx.Int32(RAWF), cnt | (vsp != fx.Int64(0)).select(fx.Int32(SPECF), zero))
    return a0, a1, nib, gmax, cwv, ev, idx, has, rawi


@flyc.jit
def enc_store(value, base, raw, tid):
    """Write code lane ``tid``'s 8 values (4 packed dwords) of the coded row at ``base`` (or its virtual wave's raw
    fallback at ``raw``) with system-coherent stores, without publishing; every lane of the hardware wave codes one
    virtual wave."""
    lane, wave = tid % 64, tid // 64
    a0, a1, nib, gmax, cwv, ev, idx, has, rawi = enc(value, lane)
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=CODED)
    rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
    buffer_ops.buffer_store(value, rr, tid * 4, mask=rawi != 0, cache_modifier=SYS)
    buffer_ops.buffer_store(Vec.from_elements([a0, a1, nib], fx.Int32), rs, tid * 3, cache_modifier=SYS)
    buffer_ops.buffer_store(gmax.to(fx.Int8), rs, HDR + 32 * wave + lane // 4, mask=lane % 4 == 0,
                            cache_modifier=SYS, offset_is_bytes=True)
    buffer_ops.buffer_store(cwv, rs, HDR // 4 + 8 * wave + 4, mask=lane == 0, cache_modifier=SYS)
    buffer_ops.buffer_store(ev.to(fx.Int16), rs, HDR + 32 * wave + 20 + 2 * (idx - 1),
                            mask=has & (rawi == 0), cache_modifier=SYS, offset_is_bytes=True)


def _dec_fast(pw, pb, hw, lane):
    """4 packed bf16 pairs of a lane from its planes and its wave's header words (no zero / exception)."""
    q4 = lane // 16
    hs = (q4 == 0).select(hw[0], (q4 == 1).select(hw[1], (q4 == 2).select(hw[2], hw[3])))
    gE = (hs >> (((lane // 4) % 4) * 8)) & 0xFF
    e16 = gE | (gE << 16)
    out = []
    for p in range_constexpr(4):
        w = pw[p // 2]
        sh = 16 * (p % 2)
        a16 = ((w >> sh) & 0xFF) | (((w >> (sh + 8)) & 0xFF) << 16)
        d16 = (pb >> (4 * p)) & 0x000F000F
        out.append(((a16 << 8) & SIGN2) | (((e16 - d16) << 7) | (a16 & 0x007F007F)))
    return out


def _dec_fix(pairs, pw, pb, hw):
    """The pairs with zeros and exceptions applied (the per-value selects)."""
    zero = fx.Int32(0)
    e0, e1, e2 = hw[5], hw[6], hw[7]
    v16 = []
    for t in range_constexpr(8):
        a = (pw[t // 4] >> (8 * (t % 4))) & 0xFF
        d = (pb >> NIBPOS[t]) & 15
        isx = (d == 15) & ((a & 0x7F) != 0)
        sl = isx.select((a & 0x7F) - 1, zero)
        h = sl // 2
        dw = (h == 0).select(e0, (h == 1).select(e1, e2))
        xv = (dw >> ((sl % 2) * 16)) & 0xFFFF
        normal = (pairs[t // 2] >> (16 * (t % 2))) & 0xFFFF
        v16.append(isx.select(xv, (d == 15).select((a & 0x80) << 8, normal)))
    return [v16[2 * p] | (v16[2 * p + 1] << 16) for p in range_constexpr(4)]


def load16(base, tx, mask, cm=SYS):
    """Thread tx's two records and its virtual wave's 8 header words of the coded row at ``base``; issue every load
    of a row before decoding; ``cm`` as ``dec16``."""
    rs = buffer_ops.create_buffer_resource_from_addr(base, num_records_bytes=CODED)
    recs = []
    for k in range_constexpr(2):
        r = Vec(buffer_ops.buffer_load(rs, tx * 6 + k * 3, vec_width=3, dtype=fx.Int32, mask=mask,
                                       cache_modifier=cm))
        recs.append([fx.Int32(r[t]) for t in range_constexpr(3)])
    hw = []
    for k in range_constexpr(2):
        h = Vec(buffer_ops.buffer_load(rs, HDR // 4 + 8 * (tx // 32) + 4 * k, vec_width=4, dtype=fx.Int32,
                                       mask=mask, cache_modifier=cm))
        hw += [fx.Int32(h[t]) for t in range_constexpr(4)]
    return recs, hw


def _dec8(rec, hw, raw, tid, mask, cm):
    """Code lane ``tid``'s 4 packed bf16 pairs. Zeros / exceptions by selects and the raw load by a lane mask: one
    hardware wave holds two virtual waves, so no wave-uniform branch."""
    pw, pb = rec[:2], rec[2]
    d = _dec_fix(_dec_fast(pw, pb, hw, tid % 64), pw, pb, hw)
    is_raw = (hw[4] & RAWF) != 0
    rr = buffer_ops.create_buffer_resource_from_addr(raw, num_records_bytes=ROWB)
    r = Vec(buffer_ops.buffer_load(rr, tid * 4, vec_width=4, dtype=fx.Int32, mask=mask & is_raw, cache_modifier=cm))
    return Vec.from_elements([is_raw.select(fx.Int32(r[t]), d[t]) for t in range_constexpr(4)], fx.Int32)


def dec16(recs, hw, raw, tx, mask, cm=SYS):
    """Thread tx's 16 values (columns 16 tx ..) as two 4-dword packs from ``load16``'s words; a raw virtual wave
    reads the row's raw slot at ``raw``. ``cm``: SYS where the rows were written in the same launch or by the peer's
    CUs; 0 (cached) for an SDMA receive region, whose lines a system-coherent load left stale in L2 for the next
    kernel's plain loads (test_handoffs.py, SDMA reduce-scatter after the coded tail)."""
    return [_dec8(recs[k], hw, raw, 2 * tx + k, mask, cm) for k in range_constexpr(2)]
