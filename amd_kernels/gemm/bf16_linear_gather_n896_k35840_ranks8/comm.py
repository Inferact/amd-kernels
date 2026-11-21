# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank mailbox buffers of the sharded projection's gather."""
import ctypes

import torch
import torch.distributed as dist
from amd_kernels.common import hip_runtime as ops

from .device import WORLD
from .ipc import exchange
from .reference import N_SHARD
from .reduce_gather import BUF_BYTES, COUNTS, MAX_ROWS

SLAB_SPLIT = 32


class _DevicePointer:
    """A raw device allocation as a tensor (``__cuda_array_interface__``)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _release(ptr, peer):
    hip = ops._hip()
    fn = hip.hipIpcCloseMemHandle if peer else hip.hipFree
    fn.argtypes = [ctypes.c_void_p]
    rc = fn(ctypes.c_void_p(ptr))
    assert rc == 0, f"HIP buffer release failed: {rc}"


class ProjComm:
    """One communicator per stream, created collectively on the eight TP ranks (a ``dist`` group with a CPU backend).

    Every call runs on the current stream, in the same order on every rank, with the same row count.
    """

    def __init__(self, rank, group, device):
        if dist.get_world_size(group) != WORLD:
            raise ValueError("eight ranks required")
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.current_device() != device.index or dist.get_rank(group) != rank:
            raise ValueError("select this rank device before creating the communicator")
        self.rank, self.group, self.device = rank, group, device
        ptr, handle = ops.allocate_shared_buffer_and_handle(BUF_BYTES)
        peers = exchange(handle, rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(WORLD)]
        torch.as_tensor(_DevicePointer(ptr, BUF_BYTES), device=device).fill_(255)
        self.buf_table = torch.tensor(self.bufs, dtype=torch.int64, device=device)
        self.counts = torch.zeros(COUNTS, dtype=torch.int32, device=device)
        self.slab = torch.empty(SLAB_SPLIT * MAX_ROWS * N_SHARD, dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def close(self):
        if not self.bufs:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for r, ptr in enumerate(self.bufs):
            _release(ptr, r != self.rank)
        self.bufs = []
