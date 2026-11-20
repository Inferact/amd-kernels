# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Flag words and device counters of the decode indexer split over two GPUs (``decode_score.py``,
``select_merge.py``; TP2, gfx950, FlyDSL).

Flags carry the call count (graph-safe, never reset): a sender drains its sc0 sc1 stores, then stores the count into
the receiver's flag word; the receiver spins until the word reaches it (the merge's entries carry the count
themselves). The count lives on the device in two words:
the scorer reads ``CNT_SCORE`` and its workgroup 0 writes the count to ``CNT_MERGE``; the merge reads ``CNT_MERGE``
and its workgroup 0 writes it back to ``CNT_SCORE``, so no launch rewrites a word its own workgroups read.
"""

import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

WORLD = 2
HEADS = 4  # index heads over both ranks; rank r owns heads 2 r, 2 r + 1
SYS = 0x11  # sc0 | sc1: system-coherent
SC1 = 0x10  # load: coherent across the XCDs' L2s
D = 128
MAX_REQS = 256  # one request per thread of the scorer's prefix sum
MAX_ROWS = MAX_REQS * 8
# flag words per rank (uncached int32): [F_IQ + r] the peer's index-q rows of request r
F_IQ = 0
FLAG_WORDS = MAX_REQS
FLAG_BYTES = FLAG_WORDS * 4
CNT_SCORE, CNT_MERGE = 0, 1
CNT_WORDS = 64


def _gptr(addr):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        fx.Int64(addr),
    )


def _flag_read(addr):
    """An atomic add of 0 at system scope (never hoisted out of a spin loop)."""
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
