# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-safe MoE combine + reduce-scatter on two MI355X GPUs.

One communicator per stream/microbatch. Construct collectively and warm up before graph capture. Buffers and counters
live for the process lifetime, as in Pools. Neither inputs nor outputs may alias the communicator's pool.
"""
import functools
import struct

import torch

from amd_kernels.common import hip_runtime as ops
from .host import Pools, WORLD, _exchange, _run_compiled
from .kernel import compile_sum2
from .lossless_x3 import CODED
from .slots4_stage_pull_sum2_add_rmsnorm_mxfp8 import (
    NORMERS, PARTS, PULLERS, STAGERS, UNROLL, compile_slots4_stage_pull_sum2_add_rmsnorm_mxfp8)
from .slots4_stage_pull import FLAG_BYTES, compile_slots4_stage_pull
from .slots4_stage_rows import compile_slots4_stage_rows
from .stage_pull import PARTS as RS_PARTS, PULLERS as RS_PULLERS, STAGERS as RS_STAGERS, UNROLL as RS_UNROLL
from .stage_pull import TICKET_BYTES, compile_rs_stage_pull
from .sum2_add_rmsnorm_mxfp8 import compile_sum2_add_rmsnorm_mxfp8

H = 6144
MAX_ROWS = 32768


@functools.cache
def _slots4_kernels(owner=False):
    return compile_slots4_stage_pull(owner=owner), compile_sum2()


@functools.cache
def _rs_kernel(*grid):
    return compile_rs_stage_pull(*grid)


class SlotsReduceScatterComm:
    """Weighted slots + shared -> this rank's BF16 rows, fixed FP32 sum order.

    ``shared`` comes in two forms, told apart by its rows: [2 m, H] (every token: each rank's partial of the shared
    expert, added to every row's slot sum) or [m, H] (owner form: this rank's whole shared expert output of its own m
    rows, added to its own rows' slot sum only; the peer's rows carry its slot sum alone). Sum order per rank p:
    ``P_p = bf16((((s0 + s1) + s2) + s3) [+ shared])``, then ``bf16((+0 + P_0) + P_1)``.

    The slot sum + shared stage overlaps the pull of the peer's sums (the own block is staged locally). Supports total
    rows divisible by two, 2..32768. The one-launch tail moves the peer's sums in ``lossless_x3``'s exact code
    (``coded``, 0.781x the bytes), the other entries BF16; no lossy communication and no atomic output adds.
    Calls on different ranks must use the same shape and order. Concurrent streams need distinct communicators. Graph
    replays and eager calls may be interleaved. ``tail`` (the one-launch tail's pullers, stagers, norm workers, parts,
    unroll) and ``rs`` (the dense reduce-scatter's pullers, stagers, parts, unroll) are fixed for the communicator's
    lifetime (their tickets count the launches).
    """

    def __init__(self, rank, cpu_group, device, max_rows=MAX_ROWS, tail=(PULLERS, STAGERS, NORMERS, PARTS, UNROLL),
                 rs=(RS_PULLERS, RS_STAGERS, RS_PARTS, RS_UNROLL), coded=True):
        assert WORLD <= max_rows <= MAX_ROWS and max_rows % WORLD == 0
        self.device = torch.device(device)
        self.max_rows = max_rows
        self.pools = Pools(rank, WORLD, cpu_group, self.device, max_rows * H * 2, sets=1)
        self.counter = torch.zeros(1, dtype=torch.int32, device=self.device)
        # the stage + pull launch's per-chunk flags (uncached, zeroed, mapped by the peer)
        ptr, handle = ops.allocate_shared_buffer_and_handle(FLAG_BYTES)
        handles = _exchange(handle, rank, WORLD, cpu_group)
        self.chunk_flags = [ptr if r == rank else ops.open_mem_handle(h) for r, h in enumerate(handles)]
        ptr, handle = ops.allocate_shared_buffer_and_handle(TICKET_BYTES)
        handles = _exchange(handle, rank, WORLD, cpu_group)
        self.rs_tickets = [ptr if r == rank else ops.open_mem_handle(h) for r, h in enumerate(handles)]
        self.rs_calls = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.stage_pull, self.sum2 = _slots4_kernels()
        self.stage_pull_owner = _slots4_kernels(True)[0]
        self.rs = _rs_kernel(*rs)
        self.coded = coded
        self.tail = _fused_tail(*tail, coded)
        self.tail_owner = _fused_tail(*tail, coded, True)
        self.tail_calls = torch.zeros(1, dtype=torch.int32, device=self.device)

    def _check_inputs(self, slots, shared):
        """-> (every token's rows, whether ``shared`` is the owner form)."""
        rows = slots.shape[0]
        assert WORLD <= rows <= self.max_rows and rows % WORLD == 0
        assert slots.shape == (rows, 4, H) and shared.shape in ((rows, H), (rows // WORLD, H)), (
            slots.shape, shared.shape)
        for x in (slots, shared):
            assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.device == self.device
        return rows, shared.shape[0] != rows

    def _stage_slots(self, slots, shared):
        """Slot sum + shared of the peer's rows into the pool, overlapped with the pull of the peer's sums of this
        rank's rows -> the bf16 staging ``[rows, H]`` of both ranks' sums of this rank's rows (rank 0's block first)."""
        rows, owner = self._check_inputs(slots, shared)
        p = self.pools
        staging = torch.empty(rows * H * 2, dtype=torch.uint8, device=self.device)
        _run_compiled(self.stage_pull_owner if owner else self.stage_pull, slots.data_ptr(), shared.data_ptr(),
                      staging.data_ptr(),
                      *p.pool_ptrs, *self.chunk_flags, self.counter, p.slot_bytes, rows, p.rank,
                      torch.cuda.current_stream(self.device))
        return staging

    def _slabs(self, staging, rows):
        block = rows // WORLD * H * 2
        return staging.data_ptr(), staging.data_ptr() + block

    def _sum(self, staging, rows, out):
        if out is None:
            out = torch.empty((rows // WORLD, H), dtype=torch.bfloat16, device=self.device)
        assert out.shape == (rows // WORLD, H) and out.dtype == torch.bfloat16
        assert out.is_contiguous() and out.device == self.device
        _run_compiled(self.sum2, out.data_ptr(), *self._slabs(staging, rows), rows // WORLD * H * 2,
                      torch.cuda.current_stream(self.device))
        return out

    def slots4_reduce_scatter(self, slots, shared, out=None):
        """Slot sum and shared add fused into BF16 reduce-scatter publication (``shared`` either form)."""
        staging = self._stage_slots(slots, shared)
        return self._sum(staging, slots.shape[0], out)

    def reduce_scatter(self, x, out=None):
        """Dense BF16 RS, same transport, graph-safe device sequence counters."""
        rows = x.shape[0]
        assert WORLD <= rows <= self.max_rows and rows % WORLD == 0
        assert x.shape == (rows, H) and x.dtype == torch.bfloat16
        assert x.is_contiguous() and x.device == self.device
        if out is None:
            out = torch.empty((rows // WORLD, H), dtype=torch.bfloat16, device=self.device)
        assert out.shape == (rows // WORLD, H) and out.dtype == torch.bfloat16
        assert out.is_contiguous() and out.device == self.device
        p = self.pools
        _run_compiled(self.rs, x.data_ptr(), out.data_ptr(), *p.pool_ptrs, *self.rs_tickets, self.counter,
                      self.rs_calls, p.slot_bytes, rows // WORLD * H * 2, p.rank,
                      torch.cuda.current_stream(self.device))
        return out

    def slots4_reduce_scatter_add_rmsnorm_mxfp8(self, slots, shared, residual, weight, eps, fused=True):
        """Slot sum + shared (either form) + BF16 RS + shard norm/quant: one launch (``fused``), else the stage + pull
        and the norm launch (the same bits; the SDMA hand-offs' tail kernel).

        Returns (row-major codes, shuffled scales, residual_out).
        """
        rows, owner = self._check_inputs(slots, shared)
        rows //= WORLD
        assert residual.shape == (rows, H) and residual.dtype == torch.bfloat16
        assert residual.is_contiguous() and residual.device == self.device
        assert weight.shape == (H,) and weight.dtype == torch.bfloat16
        assert weight.is_contiguous() and weight.device == self.device
        q = torch.empty((rows, H), dtype=torch.float8_e4m3fn, device=self.device)
        scale = torch.empty(((rows + 31) // 32 * 32, H // 32), dtype=torch.uint8, device=self.device)
        res_out = torch.empty_like(residual)
        eps_bits = struct.unpack('<i', struct.pack('<f', eps))[0]
        if fused:
            p = self.pools
            stage = torch.empty(rows * (CODED if self.coded else H * 2), dtype=torch.uint8, device=self.device)
            _run_compiled(self.tail_owner if owner else self.tail, slots.data_ptr(), shared.data_ptr(),
                          stage.data_ptr(),
                          *p.pool_ptrs, *self.chunk_flags, self.counter, self.tail_calls, p.slot_bytes,
                          slots.shape[0], p.rank, residual.data_ptr(), weight.data_ptr(), res_out.data_ptr(),
                          q.data_ptr(), scale.data_ptr(), scale.numel(), eps_bits,
                          torch.cuda.current_stream(self.device))
            return q, scale, res_out
        staging = self._stage_slots(slots, shared)
        _run_compiled(_sum_norm(), *self._slabs(staging, slots.shape[0]), residual.data_ptr(), weight.data_ptr(),
                      q.data_ptr(), scale.data_ptr(), res_out.data_ptr(), q.numel(), scale.numel(), rows, H, H,
                      eps_bits, torch.cuda.current_stream(self.device))
        return q, scale, res_out

    def warm_up(self):
        """Collective: compile all row-independent variants (both shared forms) before graph capture."""
        s = torch.zeros((2, 4, H), dtype=torch.bfloat16, device=self.device)
        r = torch.zeros((1, H), dtype=torch.bfloat16, device=self.device)
        for shared in (s[:, 0].contiguous(), r):
            self.slots4_reduce_scatter(s, shared)
            for fused in (True, False):
                self.slots4_reduce_scatter_add_rmsnorm_mxfp8(s, shared, r, r[0], 1e-6, fused)
        self.reduce_scatter(s[:, 0].contiguous())
        torch.cuda.synchronize(self.device)


@functools.cache
def _fused_tail(pullers, stagers, normers, parts, unroll, coded, owner=False):
    return compile_slots4_stage_pull_sum2_add_rmsnorm_mxfp8(pullers, stagers, normers, parts, unroll, coded, owner)


@functools.cache
def _sum_norm():
    return compile_sum2_add_rmsnorm_mxfp8()


@functools.cache
def _stage_rows(owner=False):
    """The slot sum + shared stage alone (the SDMA hand-offs' reduce-scatter source)."""
    return compile_slots4_stage_rows(owner)


@functools.cache
def _stage_rows_x3(owner=False):
    """The slot sum + shared stage with the peer's rows coded (the SDMA hand-offs' coded MoE tail)."""
    from .slots4_stage_rows_x3 import compile_slots4_stage_rows_x3
    return compile_slots4_stage_rows_x3(owner)


@functools.cache
def _sum_norm_x3():
    """The two-slab norm with the peer's slab coded."""
    from .sum2_x3_add_rmsnorm_mxfp8 import compile_sum2_x3_add_rmsnorm_mxfp8
    return compile_sum2_x3_add_rmsnorm_mxfp8()
