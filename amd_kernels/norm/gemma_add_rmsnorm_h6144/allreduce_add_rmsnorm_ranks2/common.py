# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The one-shot protocol of the two-rank all-reduce + add + Gemma RMSNorm kernels.

Every rank owns one uncached buffer mapped by its peer: its own staged partial rows (two copies, by the row's call
count parity), one 128-byte flag line per row, then the raw slots of coded rows. A stager wave writes its columns of a
batch of rows into its own copy, drains its stores and stores the batch's call count into flag (first row, wave) on the
peer; the reducer wave with the same columns polls that word in its own buffer and reads the rows from the peer's
buffer. Local writes keep the code's byte and short header stores off the link, which then carries whole-line reads.
A rank starts call n + 1 only after its call n saw the peer's call n flags, so the peer has finished reading call
n - 1: two copies suffice.

Each rank keeps two per-row call-count arrays (stagers, reducers), advanced once per call for every row the call
touches; they are equal on both ranks because the ranks make the same calls in the same order. Separate arrays: a
reducer that finishes its batch must not change the count a stager of the same rows reads.

The rank sum is fp32 with denormal operands and sums flushed to a zero of their sign, one bf16 rounding. With two
operands it is commutative: AITER's one-stage and two-stage all-reduce give the same bits at two ranks.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..add_rmsnorm.add_rmsnorm_mxfp8 import NT as CODE_NT
from ..add_rmsnorm.add_rmsnorm_mxfp8 import _code_off, _scale_off
from ..add_rmsnorm.common import H, bf16x8, pk_bf16, swap_xor1

SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores
ROWB = H * 2
MAX_ROWS = 4096
DATA_BYTES = 2 * MAX_ROWS * ROWB
LINE = 32  # flag words per row (a 128-byte line)
FLAG_OFF = DATA_BYTES
FLAG_BYTES = MAX_ROWS * LINE * 4
RAW_OFF = FLAG_OFF + FLAG_BYTES  # the coded rows' raw virtual waves (``lossless_bf16``), one slot per staged row
BUF_BYTES = RAW_OFF + DATA_BYTES
COUNT_PAD = 64  # count words past MAX_ROWS: masked rows of a last batch read them


def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def flag_store(addr, v):
    """An atomic exchange at system scope (write-through)."""
    _llvm.AtomicRMWOp(
        _llvm.AtomicBinOp.xchg,
        _gptr(addr).llvm_ptr,
        fx.Int32(v).ir_value(),
        _llvm.AtomicOrdering.monotonic,
        syncscope="",
        alignment=4,
    )


def _flag_read(addr):
    """An atomic add of 0 at system scope (a spin on it is never hoisted)."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(0).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


@flyc.jit
def spin(a, e):
    f = _flag_read(a)
    while (f - e) < 0:  # signed difference: the counts may wrap
        f = _flag_read(a)


def flag_addr(base, row, wave):
    return base + fx.Int64(FLAG_OFF) + fx.Int64((row * LINE + wave) * 4)


def data_addr(base, row, e, raw=False):
    """Row ``row``'s staged copy of call count ``e`` in ``base``'s buffer (``raw``: its raw-wave slot)."""
    return base + fx.Int64(RAW_OFF if raw else 0) + (fx.Int64(e & 1) * MAX_ROWS + fx.Int64(row)) * ROWB


def _ftz(u):
    """fp32 bits with a denormal flushed to a zero of its sign."""
    return ((u & 0x7F800000) == 0).select(u & fx.Int32(-0x80000000), u)


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def sum2(a, b):
    """bf16(ftz(ftz(a) + ftz(b))) per element of two 16 B packs (fp32, AITER's denormal flush)."""
    out = []
    for i in range_constexpr(4):
        wa, wb = fx.Int32(a[i]), fx.Int32(b[i])
        lo = _ftz((_as_f32(_ftz(wa << 16)) + _as_f32(_ftz(wb << 16))).bitcast(fx.Int32))
        hi = _ftz((_as_f32(_ftz(wa & fx.Int32(-65536))) + _as_f32(_ftz(wb & fx.Int32(-65536)))).bitcast(fx.Int32))
        out.append(pk_bf16(_as_f32(lo), _as_f32(hi)))
    return Vec.from_elements(out, fx.Int32)


def local_sum(slots, shared):
    """bf16((((s0 + s1) + s2) + s3) + shared) of 16 B packs, fp32 adds."""
    vs = [bf16x8(p) for p in slots] + [bf16x8(shared)]
    acc = vs[0]
    for v in vs[1:]:
        acc = [acc[n] + v[n] for n in range_constexpr(8)]
    return Vec.from_elements([pk_bf16(acc[2 * n], acc[2 * n + 1]) for n in range_constexpr(4)], fx.Int32)


def store_tiled(tx, rw, codes, sbs, has1, ok, qr, sr):
    """One row's codes in the (16, 16) tiles and its scale bytes (AITER layout), 8-wave lane map."""
    lane = tx % 64
    odd = (lane & 1) == 1
    p = tx // 2 + odd.select(fx.Int32(256), fx.Int32(0))
    q4 = lane % 4
    g = tx // 4 + (q4 == 1).select(fx.Int32(128), fx.Int32(0))
    # lane pairs: the even lane stores chunk 0's 16 columns, the odd chunk 1's
    recv = swap_xor1([odd.select(codes[0][n], codes[1][n]) for n in range_constexpr(2)])
    piece = Vec.from_elements(
        [
            odd.select(recv[0], codes[0][0]),
            odd.select(recv[1], codes[0][1]),
            odd.select(codes[1][0], recv[0]),
            odd.select(codes[1][1], recv[1]),
        ],
        fx.Int32,
    )
    buffer_ops.buffer_store(
        piece, qr, _code_off(rw, p) // 4, mask=(((lane & 1) == 0) | has1) & ok, cache_modifier=CODE_NT
    )
    sb = (q4 == 1).select(sbs[1], sbs[0])
    buffer_ops.buffer_store(
        sb.to(fx.Int8), sr, _scale_off(rw, g), mask=((q4 == 0) | ((q4 == 1) & has1)) & ok, offset_is_bytes=True
    )


def store_rowmajor(tx, rw, codes, sbs, has1, ok, qr, sr):
    """One row's row-major codes and its scale bytes (AITER layout), 8-wave lane map."""
    for h in range_constexpr(2):
        mask = ok if h == 0 else ok & has1
        buffer_ops.buffer_store(
            Vec.from_elements(codes[h], fx.Int32), qr, (rw * H + tx * 8 + h * 4096) // 4, mask=mask,
            cache_modifier=CODE_NT,
        )
        buffer_ops.buffer_store(
            sbs[h].to(fx.Int8), sr, _scale_off(rw, tx // 4 + h * 128), mask=mask & (tx % 4 == 0),
            offset_is_bytes=True,
        )
