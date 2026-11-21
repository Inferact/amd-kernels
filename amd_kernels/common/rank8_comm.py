# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the eight-rank hand-off: one uncached block per rank mapped on every peer, its address table and the
private epoch counters."""
import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops

WORLD = 8


def exchange(obj, rank, group):
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    for i, r in enumerate(sorted(dist.get_process_group_ranks(group=group))):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class Rank8Comm:
    """Blocks of ``nbytes`` (uncached device memory, zeroed, never freed) of the 8 ranks of ``group``: a collective,
    create it on every rank before graph capture. ``table`` is the int64[8] base addresses, ``ctr`` int32 epoch words
    (one per workgroup slot, local cached memory), ``rot`` the base addresses rotated to this rank (``rot[d]`` is rank
    ``(rank + d) % 8``'s block, ``rot[0]`` its own)."""

    def __init__(self, rank, group, device, nbytes, counters=1024):
        assert dist.get_world_size(group) == WORLD
        self.rank, self.device, self.nbytes = rank, torch.device(device), nbytes
        block, handle = ops.allocate_shared_buffer_and_handle(nbytes)
        peers = exchange(handle, rank, group)
        self.block = block
        self.bases = [block if r == rank else ops.open_mem_handle(h) for r, h in enumerate(peers)]
        self.rot = [self.bases[(rank + d) % WORLD] for d in range(WORLD)]
        self.ctr = torch.zeros(counters, dtype=torch.int32, device=self.device)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)
