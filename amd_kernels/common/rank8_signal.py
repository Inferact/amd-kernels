# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Device side of the eight-rank flag hand-off (see ``collectives/handshake_ranks8``).

Every rank owns one uncached block mapped on all peers (``rank8_comm.Rank8Comm``). A rank raises a flag by an atomic
system-scope store into the peer's block and polls its own block; the epoch (call number) rides in the flag word, so
nothing is ever reset. Call these inside a ``@flyc.kernel`` body.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T

WORLD = 8
SYS = 0x11  # sc0 | sc1: system-coherent loads, write-through stores


def gptr(addr, elem=None, align=4):
    return fx.inttoptr(
        fx.PointerType.get(elem or T.i32, address_space=fx.AddressSpace.Global, alignment=align), fx.Int64(addr)
    )


def sel8(p, a):
    """``a[p]`` for a runtime ``p`` in [0, 8) over a list of eight values (a select chain, no memory access)."""
    out = a[7]
    for i in range(6, -1, -1):
        out = (p == i).select(a[i], out)
    return out


def flag_store(addr, v):
    """An atomic system-scope store: posted write-through to the peer, no wait for its acknowledgement."""
    _llvm.StoreOp(
        fx.Int32(v).ir_value(), gptr(addr).llvm_ptr, ordering=_llvm.AtomicOrdering.monotonic, syncscope="", alignment=4
    )


def flag_load(addr):
    """An atomic system-scope load (a spin on it is never hoisted)."""
    return fx.Int32(
        _llvm.LoadOp(
            T.i32, gptr(addr).llvm_ptr, ordering=_llvm.AtomicOrdering.monotonic, syncscope="", alignment=4
        ).result
    )


@flyc.jit
def spin_until(addr, e):
    f = flag_load(addr)
    while (f - e) < 0:  # signed difference: epochs may wrap
        f = flag_load(addr)


def peer_lane(lane):
    """True for lanes 1..7: lane ``d`` serves peer ``(rank + d) % 8``."""
    return (lane >= 1) & (lane < WORLD)


def raise_flag(rot, lane, rank, e, off):
    """Lane ``d``: this rank's flag word at ``off`` of peer ``rot[d]`` becomes ``e`` (under ``peer_lane``)."""
    flag_store(sel8(lane, rot) + off + rank * 4, e)


def wait_flag(rot, lane, rank, e, off):
    """Lane ``d``: spin until the flag word of peer ``(rank + d) % 8`` in this rank's block shows ``e`` (under
    ``peer_lane``)."""
    spin_until(rot[0] + off + ((rank + lane) & 7) * 4, e)


def epoch_load(ctr_rsrc, slot):
    """This call's epoch: the counter word of ``slot`` (private to one workgroup) plus one, as a scalar."""
    return fx.Int32(buffer_ops.buffer_load(ctr_rsrc, slot, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1
