# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Agent-scope ready counters that order the stages of the one-launch expert kernel. Counters only order work: every
output element has one writer, so the result does not depend on arrival order."""
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T

from .common import raw

SC1 = 16  # cache policy: device-scope (write-through) stores
READY_STRIDE = 256  # bytes between two blocks' counters: each in its own line


def _ptr(addr_i64):
    return fx.inttoptr(fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4), fx.Int64(addr_i64))


def atomic_agent(op, addr, value):
    return fx.Int32(llvm.AtomicRMWOp(op, _ptr(addr).llvm_ptr, fx.Int32(value).ir_value(),
                                     llvm.AtomicOrdering.monotonic, syncscope="agent", alignment=4).result)


def atomic_add(addr, value):
    return atomic_agent(llvm.AtomicBinOp.add, addr, value)


def atomic_reset(addr):
    return atomic_agent(llvm.AtomicBinOp.xchg, addr, 0)


def poll_first(addr):
    """Thread 0's first read of a counter (device scope), issued early so its latency overlaps the setup."""
    rsrc = buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=8)
    off = (fx.thread_idx.x == 0).select(fx.Int32(0), fx.Int32(0x7FFFFFF0))
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.raw.ptr.buffer.atomic.add.i32",
                                        [raw(fx.Int32(0)), raw(rsrc), raw(off), raw(fx.Int32(0)), raw(fx.Int32(SC1))],
                                        [], []))


@flyc.jit
def wait_count(addr, need, consumers, got):
    """Thread 0 spins until the counter at ``addr`` reaches ``need``, then counts itself at ``addr + 4``; the last of
    ``consumers`` resets both words for the next call. The workgroup waits at a barrier."""
    if fx.thread_idx.x == 0:
        while got < need:
            rocdl.s_sleep(2)
            got = atomic_add(addr, 0)
        old = atomic_add(addr + fx.Int64(4), 1)
        if old == consumers - 1:
            atomic_reset(addr)
            atomic_reset(addr + fx.Int64(4))
    gpu.barrier()


SIGNAL_MODE = {"wait": True}


@functools.cache
def make_signal(wait=True, barrier=True):
    @flyc.jit
    def sig(addr, value=1):
        if wait:
            rocdl.s_waitcnt(vmcnt=0)
        if barrier:
            gpu.barrier()
        if fx.thread_idx.x == 0:
            atomic_add(addr, value)
    return sig


signal = make_signal(True, True)
