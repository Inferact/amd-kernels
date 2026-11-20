# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The two-rank one-shot protocol: flags, buffer layout and the bf16 rank sum.

Every rank owns one uncached buffer mapped by its peer: row slots ``[2, MAX_ROWS, 12288 B]`` (two copies by the row's
call-count parity) that it writes itself, then one 128-byte flag line per row that the peer writes (words 0-11: the
peer's partial rows of a batch staged, wave v; words 16-27: the peer's owner results of a batch staged, wave v). A
writer wave drains its stores into its own slots, then stores the batch's call count into the peer's flag word; the
reader wave with the same columns polls it in its own buffer and reads the rows from the peer's slots (the result's
scale bytes are local byte stores, the link carries whole-line reads). A rank starts call n + 1 only after its call n
saw a flag of the peer's call n, so the peer has finished reading call n - 1: two copies suffice. In one call a row's
slot holds either a partial or a result.

Each role (stagers, reducers, receivers) keeps its own per-row call counts and advances them once per call for every
row of the call, after its last read: equal on both ranks (same calls in the same order), and a role that finishes
early never changes a count another role of the same call reads.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .norm_math import H, pk_bf16

WORLD = 2
SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores
ROWB = H * 2
MAX_ROWS = 2048
LINE = 32  # flag words per row: each row's flags in their own 128-byte line
RESULT_FLAG = 16  # first result flag word of a line
DATA_BYTES = 2 * MAX_ROWS * ROWB
FLAG_OFF = DATA_BYTES
BUF_BYTES = FLAG_OFF + MAX_ROWS * LINE * 4
COUNT_WORDS = MAX_ROWS + 64  # per role; masked rows of a last batch read past MAX_ROWS


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


def flag_addr(base, row, word):
    return base + fx.Int64(FLAG_OFF) + fx.Int64((row * LINE + word) * 4)


def slot_addr(base, row, e):
    """Row ``row``'s slot of call count ``e`` in ``base``'s buffer."""
    return base + (fx.Int64(e & 1) * MAX_ROWS + fx.Int64(row)) * ROWB


def _as_f32(i):
    return fx.Int32(i).bitcast(fx.Float32)


def _ftz(u):
    """fp32 bits with a denormal flushed to a zero of its sign."""
    return ((u & 0x7F800000) == 0).select(u & fx.Int32(-0x80000000), u)


def sum2(a, b):
    """bf16(ftz(ftz(a) + ftz(b))) per element of two 16 B packs in fp32 (AITER's denormal flush; commutative)."""
    out = []
    for i in range_constexpr(4):
        wa, wb = fx.Int32(a[i]), fx.Int32(b[i])
        lo = _ftz((_as_f32(_ftz(wa << 16)) + _as_f32(_ftz(wb << 16))).bitcast(fx.Int32))
        hi = _ftz((_as_f32(_ftz(wa & fx.Int32(-65536))) + _as_f32(_ftz(wb & fx.Int32(-65536)))).bitcast(fx.Int32))
        out.append(pk_bf16(_as_f32(lo), _as_f32(hi)))
    return Vec.from_elements(out, fx.Int32)
