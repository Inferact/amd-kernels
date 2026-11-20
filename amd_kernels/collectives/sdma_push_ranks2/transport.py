# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Pushes between the two ranks of a GPU pair on the SDMA engines, ordered by the GPU (no host waits, no CUs).

Every rank owns receive regions (one IPC pool, coarse-grained) that its peer maps. ``push(region, items)`` copies this
rank's bytes into the peer's copy of a region:

1. (host, enqueued at once) ``hsa_amd_memory_async_copy_on_engine`` copies of the items on the push engines (an item
   of at least ``STRIPE_MIN`` bytes is cut into one piece per engine), every copy depending on the push's trigger
   signal (HSA, GPU_ONLY attribute: its value lives where a stream can write it);
2. an 8-byte copy of the push's sequence number into the peer's flag word of the region, after every data copy
   completed (one done signal counts them down);
3. (stream) the sequence number into the push's device word, then 0 into the trigger: the copies start once the
   stream's work ahead of the push is done.

``wait(region, seq)`` puts a ``hipStreamWaitValue64`` (flag >= seq) on the stream: the work after it sees the peer's
bytes. The command processor waits; no workgroup spins. Both ranks issue the same pushes and waits on a region in the
same order. Flag words are uncached: the peer's SDMA write lands in HBM, and the waiting command processor must not
read a stale L2 line.

Lifetimes, the caller's part: an SDMA engine reads a source after the push's stream position, outside stream order,
so a source tensor must stay alive until those reads are done. ``push(..., keep=...)`` holds the tensors;
``wait(..., release=True)`` drops the ones of every push but the latest: correct when, as in a chain of all-to-all
phases, the peer triggers its next push only after its wait for this rank's previous one (that wait saw this rank's
flag, which follows the data copies). A region's bytes are overwritten by the next push into it: the consumer of a
region must be ahead (in stream order on its rank) of that rank's next push that the peer waits for before pushing
into the region again.

Engines: ``ENGINES`` (``bench_engines.py``: one link per pair, see the README), restricted to those
``hsa_amd_memory_copy_engine_status`` reports free.
"""

import ctypes
import logging

import torch

from amd_kernels.common import hip_runtime as ops

from . import hsa_runtime as rt
from .ipc import alloc_pool, exchange, open_pool

logger = logging.getLogger(__name__)

WORLD = 2
RING = 32  # pushes in flight per transport; a slot is reused only after its copies are done
FLAG_STRIDE = 64  # bytes per (region, writer) flag word
REGION_ALIGN = 4096
ENGINES = (2,)
STRIPE_MIN = 1 << 20  # items from 1 MB are cut into one piece per engine
PIECE_ALIGN = 4096


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


class _Slot:
    __slots__ = ("trigger", "done", "fdone", "used")

    def __init__(self):
        self.trigger = rt.signal()
        self.done = rt.signal()
        self.fdone = rt.signal()
        self.used = False  # whether this slot's last push armed its signals


class SdmaPush:
    """Receive regions of ``region_bytes[k]`` bytes on both ranks, the flags, the SDMA engines and the signal ring.
    Collective: the two ranks of ``cpu_group`` (gloo) construct it at the same point. ``engines`` overrides
    ``ENGINES``."""

    def __init__(self, rank: int, cpu_group, device: torch.device, region_bytes, engines=None):
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        assert rank in (0, 1)
        self.rank, self.peer, self.device = rank, 1 - rank, device
        self.offsets, off = [], 0
        for n in region_bytes:
            self.offsets.append(off)
            off += _round_up(n, REGION_ALIGN)
        self.region_bytes = list(region_bytes)
        ptr, handle, self.pool = alloc_pool(max(off, REGION_ALIGN), device)
        nflags = _round_up(len(self.offsets) * WORLD * FLAG_STRIDE, 4096)
        flag_ptr, flag_handle = ops.allocate_shared_buffer_and_handle(nflags)
        # the sequence words the flag copies read (uncached: a stream write must reach HBM before an engine reads it)
        self.seq_words, _ = ops.allocate_shared_buffer_and_handle(RING * 8)
        peers = exchange((handle, flag_handle, rt.pci_key(device)), rank, WORLD, cpu_group)
        self.peer_pool = open_pool(peers[self.peer][0], device)
        self.flag_ptrs = [flag_ptr if r == rank else ops.open_mem_handle(h) for r, (_, h, _) in enumerate(peers)]
        agents = rt.gpu_agents()
        self.me_agent, self.peer_agent = agents[peers[rank][2]], agents[peers[self.peer][2]]
        self.engines = self._pick_engines(ENGINES if engines is None else engines)
        self.ring = [_Slot() for _ in range(RING)]
        self.pushes = 0
        self.seq = [0] * len(self.offsets)
        self._held: list[tuple] = []
        self._hip, self._hsa = rt.hip(), rt.hsa()
        logger.info("sdma_push_ranks2: rank %d, %d regions, %.0f MB, engines %s", rank, len(self.offsets),
                    off / 2**20, [e.bit_length() - 1 for e in self.engines])  # fmt: skip

    def _pick_engines(self, wanted) -> list[int]:
        mask = ctypes.c_uint32()
        rc = rt.hsa().hsa_amd_memory_copy_engine_status(self.peer_agent, self.me_agent, ctypes.byref(mask))
        free = [e for e in range(32) if mask.value >> e & 1] if rc == 0 else []
        picked = [e for e in wanted if not free or e in free] or free[:1]
        assert picked, "no SDMA engine to the peer"
        return [1 << e for e in picked]

    def region(self, k: int) -> torch.Tensor:
        """This rank's receive region ``k`` (uint8)."""
        return self.pool[self.offsets[k] : self.offsets[k] + self.region_bytes[k]]

    def _flag_off(self, k: int, writer: int) -> int:
        return (k * WORLD + writer) * FLAG_STRIDE

    def _pieces(self, items):
        """(src, offset, bytes, engine) copies: an item from ``STRIPE_MIN`` bytes cut into one piece per engine."""
        out, n_eng, i = [], len(self.engines), 0
        for src, off, n in items:
            if n_eng == 1 or n < STRIPE_MIN:
                out.append((src, off, n, self.engines[i % n_eng]))
                i += 1
                continue
            part = _round_up(-(-n // n_eng), PIECE_ALIGN)
            for j in range(n_eng):
                lo = j * part
                if lo < n:
                    out.append((src + lo, off + lo, min(part, n - lo), self.engines[j]))
        return out

    def push(self, k: int, items, stream=None, keep=()) -> int:
        """Copy ``items`` = [(peer, src address, byte offset in the peer's region k, bytes)] once the stream's preceding
        work is done; returns region k's sequence number for ``wait``."""
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self.seq[k] += 1
        seq = self.seq[k]
        slot = self.ring[self.pushes % RING]
        word = self.seq_words + (self.pushes % RING) * 8
        self.pushes += 1
        lib = self._hsa
        if slot.used:  # the slot's previous copies must be done before its signals are reset
            for s in (slot.done, slot.fdone):
                while lib.hsa_signal_wait_scacquire(s, rt.HSA_SIGNAL_CONDITION_LT, 1, 1 << 40,
                                                    rt.HSA_WAIT_STATE_ACTIVE) > 0:  # fmt: skip
                    pass
        for p, _, off, n in items:
            assert p == self.peer and 0 <= off and off + n <= self.region_bytes[k] and n > 0, (p, off, n)
        pieces = self._pieces([item[1:] for item in items])
        lib.hsa_signal_store_relaxed(slot.trigger, 1)
        trigger = (rt.Signal * 1)(slot.trigger)
        lib.hsa_signal_store_relaxed(slot.done, len(pieces))
        lib.hsa_signal_store_relaxed(slot.fdone, 1)
        dst = self.peer_pool + self.offsets[k]
        for src, off, n, engine in pieces:
            rc = lib.hsa_amd_memory_async_copy_on_engine(dst + off, self.peer_agent, src, self.me_agent, n, 1, trigger,
                                                         slot.done, engine, True)  # fmt: skip
            assert rc == 0, f"SDMA copy to the peer failed: {rc:#x}"
        dep = (rt.Signal * 1)(slot.done)
        rc = lib.hsa_amd_memory_async_copy_on_engine(self.flag_ptrs[self.peer] + self._flag_off(k, self.rank),
                                                     self.peer_agent, word, self.me_agent, 8, 1, dep, slot.fdone,
                                                     self.engines[0], True)  # fmt: skip
        assert rc == 0, f"SDMA flag copy to the peer failed: {rc:#x}"
        slot.used = True
        s = ctypes.c_void_p(stream.cuda_stream)
        assert self._hip.hipStreamWriteValue64(s, word, seq, 0) == 0
        assert self._hip.hipStreamWriteValue64(s, rt.value_pointer(slot.trigger), 0, 0) == 0
        self._held.append(tuple(keep))
        return seq

    def wait(self, k: int, seq: int, peers=None, stream=None, release: bool = True) -> None:
        """The stream waits for the peer's push ``seq`` into this rank's region k (``peers``: the ranks4 signature;
        only the peer counts)."""
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        s = ctypes.c_void_p(stream.cuda_stream)
        if peers is None or self.peer in peers:
            rc = self._hip.hipStreamWaitValue64(s, self.flag_ptrs[self.rank] + self._flag_off(k, self.peer), seq,
                                                rt.HIP_STREAM_WAIT_VALUE_GTE, 0xFFFFFFFFFFFFFFFF)  # fmt: skip
            assert rc == 0, f"hipStreamWaitValue64 failed: {rc}"
        if release:
            del self._held[:-1]

    def drain(self) -> None:
        """Host: wait until every push's copies are done, and drop the held sources."""
        lib = self._hsa
        for slot in self.ring:
            if slot.used:
                for s in (slot.done, slot.fdone):
                    while lib.hsa_signal_wait_scacquire(s, rt.HSA_SIGNAL_CONDITION_LT, 1, 1 << 40,
                                                        rt.HSA_WAIT_STATE_ACTIVE) > 0:  # fmt: skip
                        pass
        self._held.clear()
