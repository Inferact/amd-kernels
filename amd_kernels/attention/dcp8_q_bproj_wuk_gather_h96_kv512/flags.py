# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""System-scope flag words over IPC-mapped memory, and the hand-off protocol of the push all-gather.

Every rank maps the other ranks' gather buffer (uncached device memory): a flag area, then the data. A call ``e``
(0-based, counted on the device in ``state[0]``, so a captured graph replays it) runs:

1. ``free``: a workgroup of rank s stores ``e + 1`` to ``free[s]`` of all eight ranks when its kernel starts. Every
   pushing workgroup waits for all eight ``free`` words before its first remote store (the wait hides behind the
   compute). The latent variant has one data area and waits for ``e + 1`` (rank s's previous call and the consumer of
   its result, on the same stream, are done by then). The q variant keeps two data areas, call ``e`` using area
   ``e % 2``, and waits for ``e``: rank s started call ``e - 1``, so its consumer of call ``e - 2`` is done, and the
   wait never depends on this call's start skew between the ranks.
2. ``push``: stores to the eight data areas, ``s_waitcnt vmcnt(0)`` and a barrier, then one store of ``e + 1`` to the
   word ``ready[s][k]`` of each rank, k the index of the pushed tile (token tile x column block, or the pushing
   workgroup). The data area is uncached: a store reaches the peer without a flush, and the words are plain stores to
   distinct addresses, so there is no atomic to serialize.
3. ``wait``: a reader polls the ``ready[s][k]`` words of exactly the data it reads until each reaches ``e + 1``
   (a signed difference: the words are never reset), then reads with system-scope loads; the q kernel polls the words
   of all tiles it has not consumed, once per round, and reads the tiles whose words are all set.
4. ``retire``: the last workgroup of the call (a ticket in ``state[1]``, reset by it) stores ``state[0] += 1``. Every
   workgroup has read ``state[0]`` at its start, and the next call's kernels run after this one on the stream.

Adapted from gemm/bf16_concat_linear_n1536_k30720/flags.py of this repository (monotonic atomics at system scope, a
spin that is never hoisted)."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

LINE = 32
FREE_OFF = 0
READY_OFF = 4096
WORDS = 4096
DATA_OFF = READY_OFF + 8 * WORDS * 4
Q_AREA = 256 * 96 * 192 * 2  # bytes of one gathered-q buffer (rows <= 256); the q variant keeps two, by epoch parity


def _gptr(addr):
    return fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4), fx.Int64(addr))


def flag_store(addr, v):
    """A system-scope write-through store (``global_store sc0 sc1``)."""
    _llvm.StoreOp(
        fx.Int32(v).ir_value(), _gptr(addr).llvm_ptr, alignment=4, ordering=_llvm.AtomicOrdering.monotonic, syncscope=""
    )


def ticket_add(addr):
    """Atomic add at agent scope (executes in L2, far cheaper than at system scope); returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _gptr(addr).llvm_ptr,
            fx.Int32(1).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="agent",
            alignment=4,
        ).result
    )


def flag_read(addr):
    """A system-scope load (``global_load sc0 sc1``): a spin on it is never hoisted."""
    return fx.Int32(
        _llvm.LoadOp(
            T.i32, _gptr(addr).llvm_ptr, alignment=4, ordering=_llvm.AtomicOrdering.monotonic, syncscope=""
        ).result
    )


@flyc.jit
def spin(addr, expect):
    f = flag_read(addr)
    while (f - expect) < 0:
        f = flag_read(addr)


@flyc.jit
def spin_from(addr, expect, first):
    """``spin`` that starts from a value loaded earlier (``first``), so a flag already set costs no round trip here."""
    f = first
    while (f - expect) < 0:
        f = flag_read(addr)


def sel8(p, vals):
    out = vals[7]
    for i in range(6, -1, -1):
        out = (p == i).select(vals[i], out)
    return out
