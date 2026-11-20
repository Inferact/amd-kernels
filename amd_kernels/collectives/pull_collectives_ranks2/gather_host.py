# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe, lossless gather of row-major tensor fields on two ranks (``stage_pull.compile_gather_stage_pull``: one
launch per six fields)."""
import functools

import torch

from amd_kernels.common import hip_runtime as ops
from .host import Pools, WORLD, _exchange, _run_compiled
from .stage_pull import FIELDS, PART_BYTES, PARTS, PULLERS, STAGERS, TICKET_BYTES, UNROLL, compile_gather_stage_pull


@functools.cache
def _gather_kernel(*grid):
    return compile_gather_stage_pull(*grid)


class AllGatherComm:
    """Gather a list of contiguous tensor shards, preserving every byte.

    Construct collectively with the same slot capacity on both ranks. Each field has a positive byte length divisible
    by 16 and 16-byte-aligned input/output addresses. Fields occupy 256-byte-aligned regions of a slot. Outputs
    concatenate ranks along their first axis. One communicator per ordered stream/microbatch. Eager calls and graph
    replay can interleave; both ranks must issue the same shapes and operation order. Inputs and outputs must not
    alias this communicator's IPC pool. ``grid`` (pullers, stagers, parts, unroll, part bytes) is fixed for the
    communicator's lifetime (its tickets count the launches).
    """

    def __init__(self, rank, cpu_group, device, slot_bytes, grid=(PULLERS, STAGERS, PARTS, UNROLL, PART_BYTES)):
        assert 0 < slot_bytes < 2**31 and slot_bytes % 256 == 0
        self.device = torch.device(device)
        self.pools = Pools(rank, WORLD, cpu_group, self.device, slot_bytes, sets=1)
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        ptr, handle = ops.allocate_shared_buffer_and_handle(TICKET_BYTES)
        handles = _exchange(handle, rank, WORLD, cpu_group)
        self.tickets = [ptr if r == rank else ops.open_mem_handle(h) for r, h in enumerate(handles)]
        self.kernel = _gather_kernel(*grid)

    def all_gather(self, outs, ins):
        """Stage, publish and pull every field, six per launch."""
        assert len(outs) == len(ins) and len(ins) > 0
        for i in range(0, len(ins), FIELDS):
            self._launch(outs[i:i + FIELDS], ins[i:i + FIELDS])

    def _launch(self, outs, ins):
        p = self.pools
        offsets, offset = [], 0
        for out, x in zip(outs, ins):
            assert x.ndim > 0 and x.is_contiguous() and out.is_contiguous()
            assert x.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
            assert x.device == self.device and out.device == self.device and out.dtype == x.dtype
            assert tuple(out.shape) == (WORLD * x.shape[0], *x.shape[1:])
            n = x.numel() * x.element_size()
            assert n > 0 and n % 16 == 0
            offset = (offset + 255) // 256 * 256
            assert offset + n <= p.slot_bytes
            for t in (x, out):
                assert not (p.pool.data_ptr() <= t.data_ptr() < p.pool.data_ptr() + p.pool.numel())
            offsets.append((offset, n))
            offset += n
        pad = FIELDS - len(ins)
        _run_compiled(self.kernel, *(x.data_ptr() for x in ins), *[0] * pad, *(o.data_ptr() for o in outs), *[0] * pad,
                      *(off for off, _ in offsets), *[0] * pad, *(n for _, n in offsets), *[0] * pad,
                      *p.pool_ptrs, *self.tickets, self.counter, p.slot_bytes, p.rank,
                      torch.cuda.current_stream(self.device))

    def warm_up(self):
        """Collective: compile the byte-count-independent transport before capture."""
        x = torch.zeros((1, 16), device=self.device, dtype=torch.uint8)
        out = torch.empty((WORLD, 16), device=self.device, dtype=torch.uint8)
        self.all_gather([out], [x])
        torch.cuda.synchronize(self.device)
