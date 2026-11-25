# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicator and entry of the DCP8 combine of packed partials (see ``kernel.py``).

One communicator per stream. Create it on every rank before graph capture (a collective), and call
``dcp8_lse_combine_packed`` on every rank in the same order with the same row count. Inputs and output may not alias
the pools.
"""

import functools

import torch
import torch.distributed as dist

from amd_kernels.common import hip_runtime as ops
from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from .kernel import CTL_BYTES, MAX_TOKENS, compile_combine, grid_of
from .reference import D, HPR, PACK, SUB, V, WORLD


@functools.cache
def _kernel():
    return compile_combine()


def _exchange(obj, rank, group):
    out = [[None] for _ in range(WORLD)]
    out[rank][0] = obj
    for i, r in enumerate(sorted(dist.get_process_group_ranks(group=group))):
        dist.broadcast_object_list(out[i], src=r, group=group, device="cpu")
    return [o[0] for o in out]


class DcpPackedCombine:
    """Pools, control blocks and the device table of the 8 ranks (a collective: every rank of ``group`` creates one).

    The pools and control blocks are uncached device memory mapped on the peers (``hipDeviceMallocUncached`` through
    IPC), so a peer's stores are never shadowed by a stale line of this rank's L2.
    """

    def __init__(self, rank, group, device, max_tokens=MAX_TOKENS):
        assert 1 <= max_tokens <= MAX_TOKENS and dist.get_world_size(group) == WORLD
        self.rank, self.device, self.max_tokens = rank, torch.device(device), max_tokens
        self.slot_bytes = -(-WORLD * max_tokens * SUB * PACK * 2 // 256) * 256
        pool, pool_h = ops.allocate_shared_buffer_and_handle(2 * self.slot_bytes)
        ctl, ctl_h = ops.allocate_shared_buffer_and_handle(CTL_BYTES)
        peers = _exchange((pool_h, ctl_h), rank, group)
        pools = [pool if r == rank else ops.open_mem_handle(h[0]) for r, h in enumerate(peers)]
        ctls = [ctl if r == rank else ops.open_mem_handle(h[1]) for r, h in enumerate(peers)]
        self.table = torch.tensor(pools + ctls, dtype=torch.int64, device=self.device)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=group)

    def _launch(self, send, out, stream):
        m = send.shape[1]
        assert 0 < m <= self.max_tokens
        assert send.dtype == torch.bfloat16 and send.shape == (WORLD, m, SUB, PACK)
        assert send.stride(3) == 1 and send.stride(2) == PACK and send.stride(1) == SUB * PACK and send.stride(0) % 2 == 0
        assert out.dtype == torch.bfloat16 and out.numel() == m * HPR * V and out.is_contiguous()
        _run_compiled(
            _kernel(), send.data_ptr(), send.stride(0) * 2, out.data_ptr(), self.table.data_ptr(), m,
            self.rank, self.max_tokens, self.slot_bytes, grid_of(-(-m // 16)), stream,
        )


def _combine(comm, send, out):
    """Write this rank's [T, 16, 192] heads of the combined attention output into ``out`` and return it.

    ``send`` [8, T, 48, 66] bf16 is this rank's packed partials: ``send[d]`` holds the 16 heads of rank d as 48
    sub-heads of 64 values followed by the fp32 base-2 LSE (two bf16 words, empty shards already -inf).
    """
    if send.shape[1]:
        comm._launch(send, out, torch.cuda.current_stream(comm.device))
    return out


def _mode(comm, send, out):
    return (comm.max_tokens,)


def _sizes(mode):
    return [1]


def _inputs(mode, n, comm, send, out):
    dev = send.device
    return (
        comm,
        torch.zeros((WORLD, n, SUB, PACK), dtype=torch.bfloat16, device=dev),
        torch.zeros((n, HPR, V), dtype=torch.bfloat16, device=dev),
    ), {}


dcp8_lse_combine_packed = entry(
    "attention/dcp8_lse_combine_packed_h16_v192", mode=_mode, sizes=_sizes, inputs=_inputs
)(_combine)
